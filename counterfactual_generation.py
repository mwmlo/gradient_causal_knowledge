import torch
from torch import Tensor
from torch.nn import CosineSimilarity
from torchjd.aggregation import MGDA
from torchjd import backward
import numpy as np

from transformer_lens.utils import get_act_name, get_device
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from sentence_transformers.util import semantic_search, normalize_embeddings, dot_score


class Embedding:
    """
    Class for a continuous-valued vector embedding and its projection for the given model.
    """
    def __init__(self, model: HookedTransformer, vector):
        self.d_model = model.cfg.d_model
        self.embedding_matrix = model.W_E
        # Continuous-valued embedding vector
        self.vector = vector
        # Projected embedding vector (maps to model tokens)
        self.projection = nn_project(vector, self.embedding_matrix)

    def update(self, new_vector):
        self.vector = new_vector
        self.projection = nn_project(new_vector, self.embedding_matrix)


def nn_project(curr_embeds, embedding_matrix):
    """
    Return the closest embedding in the model's vocabulary.
    Taken from "Hard Prompts Made Easy": https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py#L26
    """
    with torch.no_grad():
        batch_size, seq_len, d_model = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = torch.reshape(curr_embeds, (-1, d_model))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        norm_embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, norm_embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)

        projected_embeds = embedding_matrix[nn_indices]
        projected_embeds = projected_embeds.reshape((batch_size, seq_len, d_model))

    return projected_embeds


def run_from_layer_fn(model, original_input, patch_layer, patch_input, reset_hooks_end=True):
    """
    Run the model, patching in `patch_input` at the target layer.
    """
    
    # Do not backpropagate before the target layer
    torch.set_grad_enabled(False)

    def fwd_hook(act, hook):
        torch.set_grad_enabled(True)
        patch_input.requires_grad = True
        return patch_input
    
    logits = model.run_with_hooks(
        original_input,
        fwd_hooks=[(patch_layer.name, fwd_hook)],
        reset_hooks_end=reset_hooks_end
    )
    return logits


def compute_outputs(model: HookedTransformer, contrastive_pair: list[Embedding], target_layer, target_index):
    x, y = contrastive_pair
    target_layer_name = target_layer.name

    is_target = lambda name: name == target_layer_name
    (x_logits, x_ce_loss), x_cache = \
        model.run_with_cache(x.projection, return_type="both", names_filter=is_target)
    # Treat x as the "clean" input - as such, its top token is the "correct" answer 
    answer_x_logit, answer_idx = torch.max(x_logits[0, -1])
    
    (_, y_ce_loss), y_cache = \
        model.run_with_cache(y.projection, return_type="both", names_filter=is_target)

    component_y_input = x_cache[target_layer_name].clone()
    component_y_input[:, :, target_index] = y_cache[target_layer_name][:, :, target_index]

    model.reset_hooks()
    y_logits = run_from_layer_fn(model, x.projection, target_layer, component_y_input, reset_hooks_end=False)
    answer_y_logit = y_logits[0, -1, answer_idx]

    return answer_x_logit, answer_y_logit, x_ce_loss, y_ce_loss, answer_idx


def compute_similarity(contrastive_pair):
    x, y = contrastive_pair
    cosine_similarity_loss = CosineSimilarity(dim=-1)
    return cosine_similarity_loss(x.projection, y.projection)


def counterfactuals(start_input: str, model: HookedTransformer, target_layer: HookPoint, target_index: int, iterations=100):
    # Initialise contrastive pair
    start_tokens = model.to_tokens(start_input)
    start_embed = model.embed(start_tokens)
    
    x = Embedding(model, start_embed)
    y = Embedding(model, start_embed)
    contrastive_pair = torch.tensor([x, y], requires_grad=True)

    optimizer = torch.optim.AdamW(contrastive_pair)
    aggregator = MGDA()

    # Gradient descent
    for step in range(iterations):

        answer_x_logit, answer_y_logit, x_ce_loss, y_ce_loss, answer_token = \
            compute_outputs(model, contrastive_pair, target_layer, target_index)
        
        print(f"Correct answer: {model.to_string(answer_token)}")

        output_diff = answer_x_logit - answer_y_logit
        output_grads = torch.autograd.grad(output_diff, contrastive_pair)[0]

        input_similarity = compute_similarity(contrastive_pair)
        negative_perplexity = torch.Tensor([-torch.exp(x_ce_loss), -torch.exp(y_ce_loss)])

        losses = [output_diff, output_grads, input_similarity, negative_perplexity]
        print(f"Losses: {losses}")

        optimizer.zero_grad()
        backward(losses=losses, aggregator=aggregator)
        optimizer.step()

    x, y = contrastive_pair
    return x.projection, y.projection