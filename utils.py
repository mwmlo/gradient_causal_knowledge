import matplotlib.pyplot as plt
import torch

def plot_attn(attn_results, model):
    
    bound = attn_results.abs().max()

    plt.imshow(attn_results, cmap="RdBu", vmin=-bound, vmax=bound)

    plt.xlabel("Head Index")
    plt.xticks(list(range(model.cfg.n_heads)))
    plt.ylabel("Layer")
    plt.yticks(list(range(model.cfg.n_layers)))

    plt.colorbar()
    plt.tight_layout()
    plt.show()