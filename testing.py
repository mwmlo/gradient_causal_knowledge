import pandas as pd
import numpy as np
import torch
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
from enum import Enum
from torch.utils.data import Dataset, DataLoader


class Task(Enum):
    IOI = 1
    GENDER_BIAS = 2
    GREATER_THAN = 3
    CAPITAL_COUNTRY = 4
    SVA = 5
    HYPERNYMY = 6


class TaskDataset(Dataset):
    """
    Loads datasets for circuit tasks.
    Implementation based on https://github.com/hannamw/eap-ig-faithfulness
    """

    def __init__(self, task: Task):
        filename = task.name.lower()
        self.task = task
        self.df = pd.read_csv(f"datasets/{filename}.csv")

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = None

        if self.task == Task.IOI:
            label = [row["correct_idx"], row["incorrect_idx"]]
            return row["clean"], row["corrupted_hard"], label

        if self.task == Task.GREATER_THAN:
            label = row["correct_idx"]
        elif self.task == Task.HYPERNYMY:
            answer = torch.tensor(eval(row["answers_idx"]))
            corrupted_answer = torch.tensor(eval(row["corrupted_answers_idx"]))
            label = [answer, corrupted_answer]
        elif self.task == Task.CAPITAL_COUNTRY:
            label = [row["country_idx"], row["corrupted_country_idx"]]
        elif self.task == Task.GENDER_BIAS:
            label = [row["clean_answer_idx"], row["corrupted_answer_idx"]]
        elif self.task == Task.SVA:
            label = row["plural"]
        else:
            raise ValueError(f"Got invalid task: {self.task}")

        return row["clean"], row["corrupted"], label

    def to_dataloader(self, batch_size: int):

        def collate_EAP(xs):
            clean, corrupted, labels = zip(*xs)
            clean = list(clean)
            corrupted = list(corrupted)
            if self.task != Task.HYPERNYMY:
                labels = torch.tensor(labels)
            return clean, corrupted, labels

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)


def logit_diff_metric(logits, metric_labels):
    """
    Calculate difference in logit values between the correct token and the incorrect token.
    """
    correct_index = metric_labels[:, 0]
    incorrect_index = metric_labels[:, 1]
    logits_last = logits[:, -1, :]
    batch_size = logits.size(0)
    correct_logits = logits_last[torch.arange(batch_size), correct_index]
    incorrect_logits = logits_last[torch.arange(batch_size), incorrect_index]
    return correct_logits - incorrect_logits


def greater_than_prob_diff_metric(model, logits, correct_years):
    """
    Calculate difference in logit probabilities between numbers above and below the correct year.
    """
    batch_size = logits.size(0)
    year_indices = model.to_tokens([f"{year:02d}" for year in range(100)])[
        :, 1
    ]  # Shape [100]
    logits_last = logits[:, -1, :]
    logit_probs = torch.softmax(logits_last, dim=-1)  # Shape [batch, d_vocab]
    year_probs = logit_probs[:, year_indices]  # Shape [batch, 100]

    results = torch.zeros((batch_size))
    for i, (probs, year) in enumerate(zip(year_probs, correct_years)):
        results[i] = probs[year+1:].sum() - probs[:year+1].sum()

    results = results.to(logits.device)
    return results


def run_ablated_component(
    model: HookedTransformer,
    is_attn: bool,
    layer_idx: int,
    component_idx: int,
    corrupted_cache: ActivationCache,
    metric: callable,
    metric_labels: Tensor,
    *model_args,
    **model_kwargs,
):
    """
    Run the model with the component at (layer_idx, head_idx) corrupted.
    Component can either be attention head or MLP neuron.
    """
    if is_attn:
        layer_name = get_act_name("result", layer_idx)
    else:
        layer_name = get_act_name("post", layer_idx)

    # Patch in corrupted activations
    def ablate_hook(act, hook):
        act[:, :, component_idx] = corrupted_cache[hook.name][:, :, component_idx]
        return act

    logits = model.run_with_hooks(
        *model_args, **model_kwargs, fwd_hooks=[(layer_name, ablate_hook)]
    )
    return metric(logits, metric_labels)


def test_ablated_performance(
    model: HookedTransformer,
    is_attn: bool,
    layer_idx: int,
    component_idx: int,
    task: Task,
    n_samples=100,
):
    """
    Evaluate the model's performance on a task dataset, when component at (layer_idx, component_idx) is ablated.
    """
    print(f"Test {task.name} performance with ablated {layer_idx, component_idx}")
    test_dataset = TaskDataset(task)
    test_dataloader = test_dataset.to_dataloader(batch_size=10)

    if task == Task.GREATER_THAN:
        metric = greater_than_prob_diff_metric
    else:
        metric = logit_diff_metric

    mean_performance = 0

    for i, (clean_input, corrupted_input, labels) in enumerate(test_dataloader):
        clean_tokens = model.to_tokens(clean_input)
        corrupted_tokens = model.to_tokens(corrupted_input)
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

        performance = run_ablated_component(
            model,
            is_attn,
            layer_idx,
            component_idx,
            corrupted_cache,
            metric,
            labels,
            clean_tokens,
        )

        mean_performance += performance.sum()

        if i > n_samples:
            break

    mean_performance /= len(test_dataset)

    print(f"Mean performance: {mean_performance}")
    return mean_performance


def average_correlation(x, y):
    n_samples = x.size(0)
    xs = x.reshape(n_samples, -1)
    ys = y.reshape(n_samples, -1)

    total_corr = 0
    for x, y in zip(xs, ys):
        corr = np.corrcoef(x, y)[0, 1]
        total_corr += np.abs(corr)

    avg_corr = total_corr / n_samples
    return avg_corr
