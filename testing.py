import pandas as pd
import numpy as np
import torch
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
from enum import Enum
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


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

    def to_dataloader(self, batch_size: int, shuffle: bool = False):

        def collate_EAP(xs):
            clean, corrupted, labels = zip(*xs)
            clean = list(clean)
            corrupted = list(corrupted)
            if self.task != Task.HYPERNYMY:
                labels = torch.tensor(labels)
            return clean, corrupted, labels

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_EAP)


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
    year_indices = model.to_tokens([f"{year:02d}" for year in range(100)])[:, 1]  # [100]
    logits_last = logits[:, -1, :]
    logit_probs = torch.softmax(logits_last, dim=-1)  # [batch, vocab]
    year_probs = logit_probs[:, year_indices]  # [batch, 100]

    # Create a [batch, 100] mask for indices > year
    indices = torch.arange(100, device=correct_years.device).unsqueeze(0)  # [1, 100]
    correct_years = correct_years.unsqueeze(1)  # [batch, 1]

    greater_mask = (indices > correct_years).float().to(year_probs.device)
    lesser_equal_mask = (indices <= correct_years).float().to(year_probs.device)

    greater_sum = (year_probs * greater_mask).sum(dim=1)  # [batch]
    lesser_equal_sum = (year_probs * lesser_equal_mask).sum(dim=1)  # [batch]
    
    result = greater_sum - lesser_equal_sum  # [batch]
    return result


# def greater_than_prob_diff_metric(model, logits, correct_years):
#     """
#     Calculate difference in logit probabilities between numbers above and below the correct year.
#     """
#     batch_size = logits.size(0)
#     year_indices = model.to_tokens([f"{year:02d}" for year in range(100)])[
#         :, 1
#     ]  # Shape [100]
#     logits_last = logits[:, -1, :]
#     logit_probs = torch.softmax(logits_last, dim=-1)  # Shape [batch, d_vocab]
#     year_probs = logit_probs[:, year_indices]  # Shape [batch, 100]

#     results = torch.zeros((batch_size))
#     for i, (probs, year) in enumerate(zip(year_probs, correct_years)):
#         results[i] = probs[year+1:].sum() - probs[:year+1].sum()

#     results = results.to(logits.device)
#     return results


def run_single_ablated_component(
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
        # print(act.shape, corrupted_cache[hook.name].shape)
        act[:, :, component_idx] = corrupted_cache[hook.name][:, :, component_idx]
        return act

    logits = model.run_with_hooks(
        *model_args, **model_kwargs, fwd_hooks=[(layer_name, ablate_hook)]
    )
    return metric(logits, metric_labels)


def test_single_ablated_performance(
    model: HookedTransformer,
    layer_idx: int,
    component_idx: int,
    corrupt_cache: ActivationCache,
    task: Task,
    is_attn: bool,
    n_samples=100,
):
    """
    Evaluate the model's performance on a task dataset, when component at (layer_idx, component_idx) is ablated.
    Ablate by replacing activations with those from given corrupt_cache.
    """
    print(f"Test {task.name} performance with ablated {layer_idx, component_idx}")
    test_dataset = TaskDataset(task)
    test_dataloader = test_dataset.to_dataloader(batch_size=10)

    if task == Task.GREATER_THAN:
        metric = greater_than_prob_diff_metric
    else:
        metric = logit_diff_metric

    mean_performance = 0

    for i, (clean_input, _, labels) in enumerate(test_dataloader):
        clean_tokens = model.to_tokens(clean_input)

        performance = run_single_ablated_component(
            model,
            is_attn,
            layer_idx,
            component_idx,
            corrupt_cache,
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


def run_single_amplified_component(
    model: HookedTransformer,
    is_attn: bool,
    layer_idx: int,
    component_idx: int,
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
    def amplify_hook(act, hook):
        act[:, :, component_idx] = act[:, :, component_idx] * 2
        return act

    logits = model.run_with_hooks(
        *model_args, **model_kwargs, fwd_hooks=[(layer_name, amplify_hook)]
    )
    return metric(logits, metric_labels)


def test_single_amplified_performance(
    model: HookedTransformer,
    layer_idx: int,
    component_idx: int,
    task: Task,
    is_attn: bool,
    n_samples=100,
):
    """
    Evaluate the model's performance on a task dataset, when component at (layer_idx, component_idx) is ablated.
    Ablate by replacing activations with those from given corrupt_cache.
    """
    print(f"Test {task.name} performance with ablated {layer_idx, component_idx}")
    test_dataset = TaskDataset(task)
    test_dataloader = test_dataset.to_dataloader(batch_size=10)

    if task == Task.GREATER_THAN:
        metric = greater_than_prob_diff_metric
    else:
        metric = logit_diff_metric

    mean_performance = 0

    for i, (clean_input, _, labels) in enumerate(test_dataloader):
        clean_tokens = model.to_tokens(clean_input)

        performance = run_single_amplified_component(
            model,
            is_attn,
            layer_idx,
            component_idx,
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


def run_multi_ablated_components(
    model: HookedTransformer,
    is_attn: bool,
    ablation_indices: list,
    corrupt_cache: ActivationCache,
    metric: callable,
    metric_labels: Tensor,
    input_tokens: Tensor,
):
    """
    Run the model with all components in ablation_indices corrupted.
    Component can either be attention head or MLP neuron.
    """

    # Patch in corrupted activations
    def ablate_hook(act, hook, component_idx):
        corrupt_acts = corrupt_cache[hook.name]
        if len(corrupt_acts.shape) < len(act.shape):
            corrupt_acts = corrupt_acts.unsqueeze(0)
        act[:, :, component_idx] = corrupt_acts[:, :, component_idx]
        return act

    # Attach ablation hooks at every ablation target location
    hook_tuples = []
    for layer_idx, component_idx in ablation_indices:
        if is_attn:
            hook_name = get_act_name("result", layer_idx)
        else:
            hook_name = get_act_name("post", layer_idx)

        hook_tuples.append(
            (hook_name, lambda act, hook: ablate_hook(act, hook, component_idx))
        )

    logits = model.run_with_hooks(
        input_tokens, fwd_hooks=hook_tuples
    )
    performance = metric(logits, metric_labels)
    
    return performance


def test_multi_ablated_performance(
    model: HookedTransformer,
    ablation_indices: list,
    corrupt_cache: ActivationCache,
    task: Task,
    is_attn: bool,
    n_samples=100,
):
    """
    Evaluate the model's performance on a task dataset, when component at (layer_idx, component_idx) is ablated.
    """
    print(f"Test {task.name} performance with {n_samples} samples of ablated components")
    test_dataset = TaskDataset(task)
    test_dataloader = test_dataset.to_dataloader(batch_size=1)

    if task == Task.GREATER_THAN:
        metric = greater_than_prob_diff_metric
    else:
        metric = logit_diff_metric

    # Separate ablation indices by sample
    grouped_ablation_indices = defaultdict(list)
    for sample_idx, layer_idx, component_idx in ablation_indices:
        
        if isinstance(sample_idx, Tensor):
            sample_idx = sample_idx.item()
            layer_idx = layer_idx.item()
            component_idx = component_idx.item()
        
        grouped_ablation_indices[sample_idx].append((layer_idx, component_idx))

    mean_performance = 0

    batched_dataloader = test_dataset.to_dataloader(batch_size=n_samples)
    clean_tokens_batch = model.to_tokens(next(iter(batched_dataloader))[0])

    # Each sample may have different latent components, run one sample at a time
    for i, (_, _, labels) in enumerate(test_dataloader):

        # Stop after n_samples
        if i >= n_samples or i > len(grouped_ablation_indices):
            break

        clean_tokens = clean_tokens_batch[i]

        # Attach ablation hooks at every ablation target location
        performance = run_multi_ablated_components(
            model,
            is_attn,
            grouped_ablation_indices[i],
            corrupt_cache,
            metric,
            labels,
            clean_tokens
        )

        mean_performance += performance.sum()

    mean_performance /= n_samples

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


def measure_overlap(x, y):
    """
    Calculate Jaccard score (intersection-over-union) between two batches of binary tensors.

    Args:
        x (torch.Tensor): Binary tensor of shape (batch_size, ...), dtype=torch.bool
        y (torch.Tensor): Binary tensor of same shape as x, dtype=torch.bool

    Returns:
        torch.Tensor: Jaccard scores of shape (batch_size,)
    """
    assert x.shape == y.shape, f"{x.shape} does not match {y.shape}"
    assert x.dtype == torch.bool and y.dtype == torch.bool, "Inputs must be boolean tensors"

    # Flatten all but batch dimension
    x_flat = x.view(x.shape[0], -1)
    y_flat = y.view(y.shape[0], -1)

    intersection = (x_flat & y_flat).float().sum(dim=1)
    union = (x_flat | y_flat).float().sum(dim=1)

    # Handle case where union is 0 (both are all zeros) → Jaccard = 1
    jaccard = torch.where(union == 0, torch.ones_like(union), intersection / union)

    return jaccard


def identify_outliers(x: Tensor, y: Tensor, only_collect_x_outliers: bool = False):
    # Note that x and y should have values on the same scale
    assert x.shape == y.shape, f"Inputs must have the same shape"
    assert len(x.shape) == 2, f"Must be a 2D input"
    assert len(y.shape) == 2, f"Must be a 2D input"

    if only_collect_x_outliers:
        # Collect components in x but not in y
        diff = x - y
    else:
        diff = np.abs(x - y)

    diff_std = np.std(diff.numpy())

    outliers = []
    for layer in range(x.size(0)):
        for idx in range(x.size(1)):
            if diff[layer, idx] > 1.96 * diff_std:
                if only_collect_x_outliers and diff[layer, idx] <= 0:
                    continue
                else:
                    outliers.append((layer, idx))

    return outliers
