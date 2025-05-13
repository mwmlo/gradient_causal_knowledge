import torch
import math
from enum import Enum
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from transformer_lens import HookedTransformer, ActivationCache

import os
import sys

sys.path.append("..")

from testing import logit_diff_metric
from attribution_methods import (
    integrated_gradients,
    activation_patching,
    highlight_components,
)


class AttributionMethod(Enum):
    IG_REWRITE_ORIGINAL = 1  # IG with rewrite baseline and original input
    IG_ORIGINAL_REWRITE = 2  # IG with original baseline and rewrite input
    AP_ORIGINAL_REWRITE = 3  # AP with original clean and rewrite corrupt


save_location = {
    AttributionMethod.IG_REWRITE_ORIGINAL: "results/counterfact/ig_rewrite_original.pt",
    AttributionMethod.IG_ORIGINAL_REWRITE: "results/counterfact/ig_original_rewrite.pt",
    AttributionMethod.AP_ORIGINAL_REWRITE: "results/counterfact/ap_original_rewrite.pt",
}


def run_attribution_steps(
    model: HookedTransformer,
    original_tokens: Tensor,
    rewrite_tokens: Tensor,
    answer_labels: Tensor,
    original_cache: ActivationCache,
    rewrite_cache: ActivationCache,
    original_logit_diff: Tensor,
    rewrite_logit_diff: Tensor,
):
    """
    Run three types of attribution methods on the given data samples.
    Returns a dictionary of highlighted components per attribution method, for MLP and attention heads each.
    """
    mlp_attribution_highlights = dict()
    attn_attribution_highlights = dict()

    # Run integrated gradients with original baseline and rewrite input
    ig_original_rewrite_path = save_location[AttributionMethod.IG_ORIGINAL_REWRITE]
    if os.path.exists(ig_original_rewrite_path):
        ig_original_rewrite_mlp, ig_original_rewrite_attn = torch.load(
            ig_original_rewrite_path
        )
    else:
        ig_original_rewrite_mlp, ig_original_rewrite_attn = integrated_gradients(
            model,
            original_tokens,
            original_cache,
            rewrite_cache,
            logit_diff_metric,
            answer_labels,
        )
        torch.save(
            (ig_original_rewrite_mlp, ig_original_rewrite_attn),
            ig_original_rewrite_path,
        )

    mlp_attribution_highlights[AttributionMethod.IG_ORIGINAL_REWRITE], _ = (
        highlight_components(ig_original_rewrite_mlp)
    )
    attn_attribution_highlights[AttributionMethod.IG_ORIGINAL_REWRITE], _ = (
        highlight_components(ig_original_rewrite_attn)
    )

    # Run integrated gradients with rewrite baseline and original input
    ig_rewrite_original_path = save_location[AttributionMethod.IG_REWRITE_ORIGINAL]
    if os.path.exists(ig_rewrite_original_path):
        ig_rewrite_original_mlp, ig_rewrite_original_attn = torch.load(
            ig_rewrite_original_path
        )
    else:
        ig_rewrite_original_mlp, ig_rewrite_original_attn = integrated_gradients(
            model,
            rewrite_tokens,
            rewrite_cache,
            original_cache,
            logit_diff_metric,
            answer_labels,
        )
        torch.save(
            (ig_rewrite_original_mlp, ig_rewrite_original_attn),
            ig_rewrite_original_path,
        )

    mlp_attribution_highlights[AttributionMethod.IG_REWRITE_ORIGINAL], _ = (
        highlight_components(ig_rewrite_original_mlp)
    )
    attn_attribution_highlights[AttributionMethod.IG_REWRITE_ORIGINAL], _ = (
        highlight_components(ig_rewrite_original_attn)
    )

    # Run activation patching from rewrite (corrupt) to original (clean) activations
    ap_path = save_location[AttributionMethod.AP_ORIGINAL_REWRITE]
    if os.path.exists(ap_path):
        ap_mlp, ap_attn = torch.load(ap_path)
    else:
        ap_mlp, ap_attn = activation_patching(
            model,
            original_tokens,
            original_logit_diff,
            rewrite_cache,
            rewrite_logit_diff,
            logit_diff_metric,
            answer_labels,
        )
        torch.save((ap_mlp, ap_attn), ap_path)

    mlp_attribution_highlights[AttributionMethod.AP_ORIGINAL_REWRITE], _ = (
        highlight_components(ap_mlp)
    )
    attn_attribution_highlights[AttributionMethod.AP_ORIGINAL_REWRITE], _ = (
        highlight_components(ap_attn)
    )

    return mlp_attribution_highlights, attn_attribution_highlights


def identify_target_components(highlighted_dict: dict):
    ig_rewrite_original_highlighted = highlighted_dict[
        AttributionMethod.IG_REWRITE_ORIGINAL
    ]
    ig_original_rewrite_highlighted = highlighted_dict[
        AttributionMethod.IG_ORIGINAL_REWRITE
    ]
    ap_highlighted = highlighted_dict[AttributionMethod.AP_ORIGINAL_REWRITE]

    # Identify minimal components as those with high attribution scores in both IG and AP.
    minimal_components = ig_rewrite_original_highlighted & ap_highlighted

    # Identify latent components as those with high attribution scores in only one direction of IG.
    latent_components = (
        ig_rewrite_original_highlighted ^ ig_original_rewrite_highlighted
    )

    # Get union of minimal and latent components
    return minimal_components | latent_components


def inverted_hinge_loss(output_logits, target_index):
    logit_probs = torch.softmax(output_logits)
    # Get probability of target token
    target_prob = logit_probs[target_index]
    # Get max probability of non-target tokens
    nontarget_probs = logit_probs.clone()
    nontarget_probs[target_index] = -math.inf
    max_nontarget_prob = torch.max(nontarget_probs)
    # Calculate IHL
    return 1 + target_prob - max_nontarget_prob


def optimise_edit_components(
    model: HookedTransformer,
    logits: Tensor,
    answer_index: Tensor,
    target_mlp_components: Tensor,
    target_attn_components: Tensor,
    optimiser: optim.Optimizer,
):
    """
    Uses binary tensors target_mlp_components and target_attn_components to identify which components to edit.
    """
    optimiser.zero_grad()

    # Calculate gradients to minimise IHL loss on forget dataset + next token prediction loss on retain dataset
    loss = inverted_hinge_loss(logits, answer_index) + F.cross_entropy(
        logits, answer_index
    )
    print(f"Loss: {loss}")
    loss.backward()

    # Mask out gradients at non-target components
    for layer_idx in range(model.cfg.n_layers):
        # Attention components: W_K, W_Q, W_V, W_O matrices
        # Match attention weight shape [n_heads, d_model, d_head] or [n_heads, d_head, d_model]
        layer_attn_weight_mask = target_attn_components[layer_idx].view(
            model.cfg.n_heads, 1, 1
        )
        # Match attention bias shape [n_heads, d_head]
        layer_attn_bias_mask = target_attn_components[layer_idx].view(
            model.cfg.n_heads, 1
        )

        model.blocks[layer_idx].attn.W_K.grad *= layer_attn_weight_mask  # shape []
        model.blocks[layer_idx].attn.b_K.grad *= layer_attn_bias_mask

        model.blocks[layer_idx].attn.W_Q.grad *= layer_attn_weight_mask
        model.blocks[layer_idx].attn.b_Q.grad *= layer_attn_bias_mask

        model.blocks[layer_idx].attn.W_V.grad *= layer_attn_weight_mask
        model.blocks[layer_idx].attn.b_V.grad *= layer_attn_bias_mask

        model.blocks[layer_idx].attn.W_O.grad *= layer_attn_weight_mask
        # Attention output biases of shape [d_model,] - no need to mask on specific head

        # MLP neuron components: W_in, W_out matrices
        layer_mlp_mask = target_mlp_components[layer_idx]  # shape [d_mlp,]
        model.blocks[layer_idx].mlp.W_in.grad *= layer_mlp_mask.view(
            1, model.cfg.d_mlp
        )  # shape [d_model, d_mlp]
        model.blocks[layer_idx].mlp.W_out.grad *= layer_mlp_mask.view(
            model.cfg.d_mlp, 1
        )  # shape [d_mlp, d_model]
        model.blocks[layer_idx].mlp.b_in.grad *= layer_mlp_mask  # shape [d_mlp,]
        # MLP output biases of shape [d_model,] - no need to mask on specific neuron

    # Update weights using optimiser
    optimiser.step()


def edit_model(
    model: HookedTransformer,
    original_prompt: str,
    rewrite_prompt: str,
    answer_labels: Tensor,
    n_epochs=5,
):
    original_tokens = model.to_tokens(original_prompt)
    rewrite_tokens = model.to_tokens(rewrite_prompt)

    original_logits, original_cache = model.run_with_cache(original_tokens)
    original_logit_diff = logit_diff_metric(original_logits, answer_labels)

    rewrite_logits, rewrite_cache = model.run_with_cache(rewrite_tokens)
    rewrite_logit_diff = logit_diff_metric(rewrite_logits, answer_labels)

    # LOCALISATION STAGE

    mlp_highlighted, attn_highlighted = run_attribution_steps(
        model,
        original_tokens,
        rewrite_tokens,
        answer_labels,
        original_cache,
        rewrite_cache,
        original_logit_diff,
        rewrite_logit_diff,
    )

    target_mlp = identify_target_components(model, mlp_highlighted)
    target_attn = identify_target_components(model, attn_highlighted)

    # EDITING STAGE

    relevant_parameters = [
        p for name, p in model.named_parameters() if "attn" in name or "mlp" in name
    ]
    optimiser = optim.Adam(relevant_parameters, lr=2e-4)

    for _ in range(n_epochs):
        logits = model(original_tokens)
        answer_index = answer_labels[:, 1]  # Aim for rewritten answer
        optimise_edit_components(
            model, logits, answer_index, target_mlp, target_attn, optimiser
        )

    return model
