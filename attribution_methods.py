import torch
import pandas as pd
import numpy as np

from functools import partial
from typing import Optional

from captum.attr import LayerIntegratedGradients

from transformer_lens.utils import get_act_name, get_device
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

import seaborn as sns
import matplotlib.pyplot as plt

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


def compute_layer_to_output_attributions(model, original_input, layer_input, layer_baseline, target_layer, prev_layer, metric, metric_labels):
    n_samples = original_input.size(0)
    # Take the model starting from the target layer
    forward_fn = lambda x: run_from_layer_fn(model, original_input, prev_layer, x, metric, metric_labels)
    # Attribute to the target_layer's output
    ig_embed = LayerIntegratedGradients(forward_fn, target_layer, multiply_by_inputs=True)
    attributions, approximation_error = ig_embed.attribute(inputs=layer_input,
                                                    baselines=layer_baseline, 
                                                    internal_batch_size=n_samples,
                                                    attribute_to_layer_input=False,
                                                    return_convergence_delta=True)
    print(f"\nError (delta) for {target_layer.name} attribution: {approximation_error}")
    return attributions


def integrated_gradients(model: HookedTransformer, clean_tokens: torch.Tensor, clean_cache: ActivationCache, corrupted_cache: ActivationCache, metric: callable, metric_labels):
    n_samples = clean_tokens.size(0)
    
    # Gradient attribution for neurons in MLP layers
    mlp_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.d_mlp)
    # Gradient attribution for attention heads
    attn_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.n_heads)

    # Calculate integrated gradients for each layer
    for layer in range(model.cfg.n_layers):

        # Gradient attribution on heads
        hook_name = get_act_name("result", layer)
        target_layer = model.hook_dict[hook_name]
        prev_layer_hook = get_act_name("z", layer)
        prev_layer = model.hook_dict[prev_layer_hook]

        layer_clean_input = clean_cache[prev_layer_hook]
        layer_corrupt_input = corrupted_cache[prev_layer_hook]

        # Shape [batch, seq_len, d_head, d_model]
        attributions = compute_layer_to_output_attributions(
            model, clean_tokens, layer_corrupt_input, layer_clean_input, target_layer, prev_layer, metric, metric_labels)
        print(attributions.shape)
        # Calculate attribution score based on mean over each embedding, for each token
        per_token_score = attributions.mean(dim=3)
        score = per_token_score.mean(dim=1)
        attn_results[:, layer] = score

        # Gradient attribution on MLP neurons
        hook_name = get_act_name("post", layer)
        target_layer = model.hook_dict[hook_name]
        prev_layer_hook = get_act_name("mlp_in", layer)
        prev_layer = model.hook_dict[prev_layer_hook]

        layer_clean_input = clean_cache[prev_layer_hook]
        layer_corrupt_input = corrupted_cache[prev_layer_hook]
        
        # Shape [batch, seq_len, d_model]
        attributions = compute_layer_to_output_attributions(
            model, clean_tokens, layer_corrupt_input, layer_clean_input, target_layer, prev_layer, metric, metric_labels)
        score = attributions.mean(dim=1)
        mlp_results[:, layer] = score

    return mlp_results, attn_results


def patch_hook(activations: torch.Tensor, hook: HookPoint, cache: ActivationCache, idx: int):
    # Replace the activations for the target neuron with activations from the cached run.
    cached_activations = cache[hook.name]
    activations[:, :, idx] = cached_activations[:, :, idx]
    return activations


def activation_patching(model: HookedTransformer, clean_tokens: torch.Tensor, clean_cache: ActivationCache, clean_logit_diff, corrupted_cache: ActivationCache, corrupted_logit_diff, metric: callable, metric_labels):
    n_samples = clean_tokens.size(0)
    
    mlp_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.d_mlp)
    attn_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.n_heads)

    baseline_diff = clean_logit_diff - corrupted_logit_diff

    for layer in range(model.cfg.n_layers):
        # Activation patching on heads
        print(f"Activation patching on attention heads in layer {layer}")
        for head in range(model.cfg.n_heads):
            hook_name = get_act_name("result", layer)
            temp_hook = lambda act, hook: patch_hook(act, hook, corrupted_cache, head)

            with model.hooks(fwd_hooks=[(hook_name, temp_hook)]):
                patched_logits = model(clean_tokens)

            patched_logit_diff = metric(patched_logits, metric_labels).detach()
            # Normalise result by clean and corrupted logit difference
            attn_results[:, layer, head] = (patched_logit_diff - clean_logit_diff) / baseline_diff

        # Activation patching on MLP neurons
        print(f"Activation patching on MLP in layer {layer}")
        for neuron in range(model.cfg.d_mlp):
            hook_name = get_act_name("post", layer)
            temp_hook = lambda act, hook: patch_hook(act, hook, corrupted_cache, neuron)
            
            with model.hooks(fwd_hooks=[(hook_name, temp_hook)]):
                patched_logits = model(clean_tokens)

            patched_logit_diff = metric(patched_logits, metric_labels).detach()
            # Normalise result by clean and corrupted logit difference
            mlp_results[:, layer, neuron] = (patched_logit_diff - clean_logit_diff) / baseline_diff

    return mlp_results, attn_results
    