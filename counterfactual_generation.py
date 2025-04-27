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
    def __init__(self, model: HookedTransformer, vector: Tensor):
        self.d_model = model.cfg.d_model
        self.embedding_matrix = model.W_E
        # Continuous-valued embedding vector
        self.vector = vector.clone().detach().requires_grad_(True)
        # Projected embedding vector (maps to model tokens)
        self.projection = nn_project(vector, self.embedding_matrix)

    def update(self, new_vector):
        self.vector.data = new_vector
        self.projection = nn_project(new_vector, self.embedding_matrix)


def nn_project(curr_embeds, embedding_matrix):
    """
    Return the closest embedding in the model's vocabulary.
    Taken from "Hard Prompts Made Easy": https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py#L26
    """
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

    def fwd_hook(act, hook):
        patch_input.requires_grad = True
        return patch_input
    
    logits, loss = model.run_with_hooks(
        original_input,
        return_type="both",
        fwd_hooks=[(patch_layer.name, fwd_hook)],
        reset_hooks_end=reset_hooks_end
    )
    return logits, loss


def map_projection_to_string(model: HookedTransformer, projection_embedding: Tensor):
    logits = model.unembed(projection_embedding) # Shape [batch, seq_len, d_vocab]
    tokens = torch.argmax(logits, dim=-1) # Shape [batch, seq_len]
    return model.to_string(tokens)


def compute_outputs(model: HookedTransformer, contrastive_pair: list[Embedding], target_layer, target_index):
    x, y = contrastive_pair
    target_layer_name = target_layer.name

    x_input_str = map_projection_to_string(model, x.projection)
    y_input_str = map_projection_to_string(model, y.projection)

    print(x.projection.shape)

    (x_logits, x_ce_loss), x_cache = \
        model.run_with_cache(x_input_str, return_type="both", names_filter=[target_layer_name], reset_hooks_end=False)

    print(x_logits.shape)
    print("1", torch.autograd.grad(x_logits.mean(), x.projection, retain_graph=True))

    print("2", x_logits.grad_fn)

    # Treat x as the "clean" input - as such, its top token is the "correct" answer 
    answer_x_logit, answer_idx = x_logits[0, -1].max(dim=-1)
    
    _, y_cache = model.run_with_cache(y_input_str, return_type="both", names_filter=[target_layer_name])

    component_y_input = x_cache[target_layer_name].clone()
    component_y_input[:, :, target_index] = y_cache[target_layer_name][:, :, target_index]

    y_logits, y_ce_loss = run_from_layer_fn(model, x_input_str, target_layer, component_y_input, reset_hooks_end=False)
    answer_y_logit = y_logits[0, -1, answer_idx]

    return answer_x_logit, answer_y_logit, x_ce_loss, y_ce_loss, answer_idx


def compute_similarity(contrastive_pair):
    x, y = contrastive_pair
    cosine_similarity_loss = CosineSimilarity(dim=-1)
    return cosine_similarity_loss(x.projection, y.projection)


def counterfactuals(start_input: str, model: HookedTransformer, target_layer: HookPoint, target_index: int, iterations=100):
    # Explicitly calculate and expose the result for each attention head
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)

    # Initialise contrastive pair
    start_embed = model.input_to_embed(start_input)[0]
    print(start_embed.shape)

    torch.set_grad_enabled(True)
    
    x = Embedding(model, start_embed)
    y = Embedding(model, start_embed)
    # Optimise over the raw vector values, not the projections
    contrastive_embeddings = (x, y)
    optimizer = torch.optim.AdamW([x.vector, y.vector])
    aggregator = MGDA()

    print("0", x.projection.grad_fn)

    # Discrete gradient descent
    for step in range(iterations):
        answer_x_logit, answer_y_logit, x_ce_loss, y_ce_loss, answer_token = \
            compute_outputs(model, contrastive_embeddings, target_layer, target_index)
        
        print(answer_x_logit.grad_fn)
        print(answer_y_logit.grad_fn)

        print(f"Correct answer: {model.to_string(answer_token)}")

        output_diff = answer_x_logit - answer_y_logit

        print(f"Output diff: {output_diff}")

        output_grads = torch.autograd.grad(output_diff, [x.projection, y.projection], retain_graph=True)

        input_similarity = compute_similarity(contrastive_embeddings)
        negative_perplexity = torch.Tensor([-torch.exp(x_ce_loss), -torch.exp(y_ce_loss)])

        losses = [output_diff, output_grads, input_similarity, negative_perplexity]
        print(f"Losses: {losses}")

        optimizer.zero_grad()
        # Calculate gradients with respect to projected embeddings
        backward(tensors=losses, aggregator=aggregator, inputs=[x.projection, y.projection])
        # Apply gradient to continuous embeddings
        optimizer.step()

    x, y = contrastive_embeddings
    x_cf = map_projection_to_string(x)
    y_cf = map_projection_to_string(y)
    model.reset_hooks()
    return x_cf, y_cf