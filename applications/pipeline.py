import torch
import math
from enum import Enum
from torch import Tensor
import torch.optim as optim
from transformer_lens import HookedTransformer, ActivationCache
from ..testing import logit_diff_metric
from ..attribution_methods import (
    integrated_gradients,
    activation_patching,
    highlight_components,
)


class AttributionMethod(Enum):
    IG_REWRITE_ORIGINAL = 1  # IG with rewrite baseline and original input
    IG_ORIGINAL_REWRITE = 2  # IG with original baseline and rewrite input
    AP_ORIGINAL_REWRITE = 3  # AP with original clean and rewrite corrupt


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
    ig_original_rewrite_mlp, ig_original_rewrite_attn = integrated_gradients(
        model,
        original_tokens,
        original_cache,
        rewrite_cache,
        logit_diff_metric,
        answer_labels,
    )
    mlp_attribution_highlights[AttributionMethod.IG_ORIGINAL_REWRITE], _ = (
        highlight_components(ig_original_rewrite_mlp)
    )
    attn_attribution_highlights[AttributionMethod.IG_ORIGINAL_REWRITE], _ = (
        highlight_components(ig_original_rewrite_attn)
    )

    # Run integrated gradients with rewrite baseline and original input
    ig_rewrite_original_mlp, ig_rewrite_original_attn = integrated_gradients(
        model,
        rewrite_tokens,
        rewrite_cache,
        original_cache,
        logit_diff_metric,
        answer_labels,
    )
    mlp_attribution_highlights[AttributionMethod.IG_REWRITE_ORIGINAL], _ = (
        highlight_components(ig_rewrite_original_mlp)
    )
    attn_attribution_highlights[AttributionMethod.IG_REWRITE_ORIGINAL], _ = (
        highlight_components(ig_rewrite_original_attn)
    )

    # Run activation patching from rewrite (corrupt) to original (clean) activations
    ap_mlp, ap_attn = activation_patching(
        model,
        original_tokens,
        original_logit_diff,
        rewrite_cache,
        rewrite_logit_diff,
        logit_diff_metric,
        answer_labels,
    )
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


def inverted_hinge_loss(output_logits, target):
    logit_probs = torch.softmax(output_logits)
    # Get probability of target token
    target_prob = logit_probs[target]
    # Get max probability of non-target tokens
    nontarget_probs = logit_probs.clone()
    nontarget_probs[target] = -math.inf
    max_nontarget_prob = torch.max(nontarget_probs)
    # Calculate IHL
    return 1 + target_prob - max_nontarget_prob


def edit_components(model: HookedTransformer, target_mlp_components: Tensor, target_attn_components: Tensor):
    # TODO: Only fine tune target components
    target_indices = target_components.nonzero()
    optimizer = optim.AdamW()
    # TODO: Fine-tune to minimise IHL loss on forget dataset + next token prediction loss on retain dataset
    pass


def edit_model(
    model: HookedTransformer,
    original_prompt: str,
    rewrite_prompt: str,
    answer_labels: Tensor,
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
