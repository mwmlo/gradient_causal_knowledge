from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from sklearn.preprocessing import MaxAbsScaler
from enum import Enum
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import numpy as np

class Task(Enum):
    IOI = 1
    GENDER_BIAS = 2
    GREATER_THAN = 3
    CAPITAL_COUNTRY = 4
    SVA = 5
    HYPERNYMY = 6

# Implementation of dataset loader based on https://github.com/hannamw/eap-ig-faithfulness

def collate_EAP(xs, task: Task):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    if task != Task.HYPERNYMY:
        labels = torch.tensor(labels)
    return clean, corrupted, labels

class TaskDataset(Dataset):
    def __init__(self, task: Task):
        filename = task.name.lower()
        self.task = task
        self.df = pd.read_csv(f'datasets/{filename}.csv')

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
            label = [row['correct_idx'], row['incorrect_idx']]
            return row['clean'], row['corrupted_hard'], label
        
        if self.task == Task.GREATER_THAN:
            label = row['correct_idx']
        elif self.task == Task.HYPERNYMY:
            answer = torch.tensor(eval(row['answers_idx']))
            corrupted_answer = torch.tensor(eval(row['corrupted_answers_idx']))
            label = [answer, corrupted_answer]
        elif self.task == Task.CAPITAL_COUNTRY:
            label = [row['country_idx'], row['corrupted_country_idx']]
        elif self.task == Task.GENDER_BIAS:
            label = [row['clean_answer_idx'], row['corrupted_answer_idx']]
        elif self.task == Task.SVA:
            label = row['plural']
        else:
            raise ValueError(f'Got invalid task: {self.task}')
        
        return row['clean'], row['corrupted'], label
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=partial(collate_EAP, task=self.task))


def logit_diff_metric(logits, metric_labels,):
    correct_index = metric_labels[:, 0]
    incorrect_index = metric_labels[:, 1]
    logits_last = logits[:, -1, :]
    batch_size = logits.size(0)
    correct_logits = logits_last[torch.arange(batch_size), correct_index]
    incorrect_logits = logits_last[torch.arange(batch_size), incorrect_index]
    return correct_logits - incorrect_logits


def run_from_layer_fn(model, original_input, patch_layer, patch_output, metric, metric_labels, reset_hooks_end=True):
    def fwd_hook(act, hook):
        assert patch_output.shape == act.shape, f"Patch shape {patch_output.shape} doesn't match activation shape {act.shape}"
        return patch_output

    logits = model.run_with_hooks(
        original_input,
        fwd_hooks=[(patch_layer.name, fwd_hook)],
        reset_hooks_end=reset_hooks_end,
    )
    
    diff = metric(logits, metric_labels)
    return diff


def plot_attn(attn_results, model, bound=None):
    
    if bound is None:
        bound = attn_results.abs().max()

    plt.imshow(attn_results, cmap="RdBu", vmin=-bound, vmax=bound)

    plt.xlabel("Head Index")
    plt.xticks(list(range(model.cfg.n_heads)))
    plt.ylabel("Layer")
    plt.yticks(list(range(model.cfg.n_layers)))

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_attn_comparison(ig_attn_results, ap_attn_results, task: Task, model: HookedTransformer):

    n_results = ig_attn_results.size(0)
    assert n_results == ap_attn_results.size(0)

    for i in range(n_results):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        bound = ig_attn_results.abs().max()
        im = ax1.imshow(ig_attn_results[i].detach(), cmap='coolwarm', vmin=-bound, vmax=bound)
        ax1.set_title(f"Integrated Gradients ({i})")

        ax1.set_xlabel("Head Index")
        ax1.set_xticks(list(range(model.cfg.n_heads)))
        ax1.set_ylabel("Layer")
        ax1.set_yticks(list(range(model.cfg.n_layers)))
        plt.colorbar(im, ax=ax1)

        bound = ap_attn_results.abs().max()
        im = ax2.imshow(ap_attn_results[i].detach(), cmap='coolwarm', vmin=-bound, vmax=bound)
        ax2.set_title(f"Activation Patching ({i})")

        ax2.set_xlabel("Head Index")
        ax2.set_xticks(list(range(model.cfg.n_heads)))
        ax2.set_ylabel("Layer")
        ax2.set_yticks(list(range(model.cfg.n_layers)))
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.show()


def plot_correlation(ig_mlp_scores, ap_mlp_scores, ig_attn_scores, ap_attn_scores, task: Task):

    n_results = ig_mlp_scores.size(0)
    assert n_results == ap_mlp_scores.size(0)

    for i in range(n_results):

        x1 = ig_mlp_scores[i].flatten().numpy()
        y1 = ap_mlp_scores[i].flatten().numpy()

        x2 = ig_attn_scores[i].flatten().numpy()
        y2 = ap_attn_scores[i].flatten().numpy()

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        
        sns.regplot(x=x1, y=y1, ax=ax1)
        ax1.set_xlabel("Integrated Gradients Attribution Scores")
        ax1.set_ylabel("Activation Patching Attribution Scores")
        ax1.set_title(f"{task.name} MLP Attribution Scores ({i})")

        sns.regplot(x=x2, y=y2, ax=ax2)
        ax2.set_xlabel("Integrated Gradients Attribution Scores")
        ax2.set_ylabel("Activation Patching Attribution Scores")
        ax2.set_title(f"{task.name} Attention Heads Attribution Scores ({i})")

        plt.tight_layout()
        plt.show()

        print(f"Correlation coefficient for neurons: {np.corrcoef(x1, y1)[0, 1]}")
        print(f"Correlation coefficient for attention heads: {np.corrcoef(x2, y2)[0, 1]}")


def plot_mean_diff(ig_scores, ap_scores, title=None):

    x = ig_scores.flatten().numpy()
    y = ap_scores.flatten().numpy()

    # Mean difference plot with scaled data

    scaled_ig_scores = MaxAbsScaler().fit_transform(x.reshape(-1, 1))
    scaled_ap_scores = MaxAbsScaler().fit_transform(y.reshape(-1, 1))

    mean = np.mean([scaled_ig_scores, scaled_ap_scores], axis=0)
    diff = scaled_ap_scores - scaled_ig_scores
    md = np.mean(diff) # Mean of the difference
    sd = np.std(diff, axis=0) # Standard deviation of the difference

    sns.regplot(x=mean, y=diff, fit_reg=True, scatter=True)
    plt.axhline(md, color='gray', linestyle='--', label="Mean difference")
    plt.axhline(md + 1.96*sd, color='pink', linestyle='--', label="1.96 SD of difference")
    plt.axhline(md - 1.96*sd, color='lightblue', linestyle='--', label="-1.96 SD of difference")
    plt.xlabel("Mean of attribution scores")
    plt.ylabel("Difference (activation patching - integrated gradients)")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()