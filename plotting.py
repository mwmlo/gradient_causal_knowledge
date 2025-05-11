import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sklearn.preprocessing import MaxAbsScaler
import seaborn as sns
import numpy as np

from testing import Task


def plot_attn(attn_results, model: HookedTransformer, bound=None):

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


def plot_attn_comparison(
    attn_a,
    attn_b,
    model: HookedTransformer,
    attn_a_title="Integrated Gradients",
    attn_b_title="Activation Patching",
):

    n_results = attn_a.size(0)
    assert n_results == attn_b.size(0)

    for i in range(n_results):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        bound = attn_a.abs().max()
        im = ax1.imshow(attn_a[i].detach(), cmap="RdBu", vmin=-bound, vmax=bound)
        ax1.set_title(f"{attn_a_title} ({i})")

        ax1.set_xlabel("Head Index")
        ax1.set_xticks(list(range(model.cfg.n_heads)))
        ax1.set_ylabel("Layer")
        ax1.set_yticks(list(range(model.cfg.n_layers)))
        plt.colorbar(im, ax=ax1)

        bound = attn_b.abs().max()
        im = ax2.imshow(attn_b[i].detach(), cmap="RdBu", vmin=-bound, vmax=bound)
        ax2.set_title(f"{attn_b_title} ({i})")

        ax2.set_xlabel("Head Index")
        ax2.set_xticks(list(range(model.cfg.n_heads)))
        ax2.set_ylabel("Layer")
        ax2.set_yticks(list(range(model.cfg.n_layers)))
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.show()


def plot_correlation(x, y, x_label: str, y_label: str, title: str):
    assert x.size(0) == y.size(0)
    n_results = x.size(0)

    for i in range(n_results):
        xi = x[i].flatten().numpy()
        yi = y[i].flatten().numpy()
        sns.regplot(x=xi, y=yi)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"{title} ({i})")

        plt.tight_layout()
        plt.show()

        print(f"Correlation coefficient: {np.corrcoef(xi, yi)[0, 1]}")


def plot_correlation_comparison(
    ig_mlp_scores, ap_mlp_scores, ig_attn_scores, ap_attn_scores, task: Task
):

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
        print(
            f"Correlation coefficient for attention heads: {np.corrcoef(x2, y2)[0, 1]}"
        )


def plot_bar_chart(key_values: dict, xlabel: str, ylabel: str, title: str):
    categories = list(key_values.keys())
    values = list(key_values.values())

    plt.bar(categories, values)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_mean_diff(ig_scores, ap_scores, title=None):

    x = ig_scores.flatten().numpy()
    y = ap_scores.flatten().numpy()

    # Mean difference plot with scaled data

    scaled_ig_scores = MaxAbsScaler().fit_transform(x.reshape(-1, 1))
    scaled_ap_scores = MaxAbsScaler().fit_transform(y.reshape(-1, 1))

    mean = np.mean([scaled_ig_scores, scaled_ap_scores], axis=0)
    diff = scaled_ap_scores - scaled_ig_scores
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    sns.regplot(x=mean, y=diff, fit_reg=True, scatter=True)
    plt.axhline(md, color="gray", linestyle="--", label="Mean difference")
    plt.axhline(
        md + 1.96 * sd, color="pink", linestyle="--", label="1.96 SD of difference"
    )
    plt.axhline(
        md - 1.96 * sd,
        color="lightblue",
        linestyle="--",
        label="-1.96 SD of difference",
    )
    plt.xlabel("Mean of attribution scores")
    plt.ylabel("Difference (activation patching - integrated gradients)")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
