import torch
from torch import Tensor
from torch.nn import CosineSimilarity
from torchjd.aggregation import MGDA, UPGrad
from torchjd import backward
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_device

import evaluate

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
        self.projection = None

    def update_projection(self):
        # Update projection value based on current embedding vector
        self.projection = nn_project(self.vector, self.embedding_matrix).detach().requires_grad_(True)


def nn_project(curr_embeds, embedding_matrix):
    """
    Return the closest embedding in the model's vocabulary.
    Adapted from "Hard Prompts Made Easy": https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py#L26
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


def run_model_with_cache(model: HookedTransformer, target_layer_name, *model_args, **model_kwargs):
    cache = {}
    
    def save_acts_hook_with_grad(act, hook):
        cache[hook.name] = act  # No detach!
        return act
    
    with model.hooks(fwd_hooks=[(target_layer_name, save_acts_hook_with_grad)]):
        logits, loss = model.forward(*model_args, **model_kwargs, return_type="both")

    return logits, loss, cache


def run_from_layer_fn(model: HookedTransformer, original_input, patch_layer, patch_input, reset_hooks_end=True, start_at_layer=None, tokens=None):
    """
    Run the model, patching in `patch_input` at the target layer.
    """
    def fwd_hook(act, hook):
        # Keep patch_input in the computational graph
        return patch_input + 0 * act
    
    with model.hooks(fwd_hooks=[(patch_layer.name, fwd_hook)], reset_hooks_end=reset_hooks_end):
        logits, loss = model.forward(original_input, start_at_layer=start_at_layer, return_type="both", tokens=tokens)
    
    return logits, loss


def map_projection_to_string(model: HookedTransformer, projection_embedding: Tensor):
    with torch.no_grad():
        logits = model.unembed(projection_embedding) # Shape [batch, seq_len, d_vocab]
        tokens = torch.argmax(logits, dim=-1) # Shape [batch, seq_len]
        return tokens, model.to_string(tokens)


def compute_outputs(model: HookedTransformer, contrastive_pair: list[Embedding], target_layer, target_index):
    x, y = contrastive_pair
    target_layer_name = target_layer.name

    x_tokens, _ = map_projection_to_string(model, x.projection)
    y_tokens, _ = map_projection_to_string(model, y.projection)

    x_start_residual, _ = model.get_residual(x.projection, 0)
    x_logits, x_loss, x_cache = run_model_with_cache(model, target_layer_name, x_start_residual, start_at_layer=0, tokens=x_tokens)

    # Treat x as the "clean" input - as such, its top token is the "correct" answer 
    answer_x_logit, answer_x_idx = x_logits[0, -1].max(dim=-1)
    
    y_start_residual, _ = model.get_residual(y.projection, 0)
    _, y_loss, y_cache = run_model_with_cache(model, target_layer_name, y_start_residual, start_at_layer=0, tokens=y_tokens)

    component_y_input = x_cache[target_layer_name].clone()
    component_y_input[:, :, target_index] = y_cache[target_layer_name][:, :, target_index]

    y_logits, _ = run_from_layer_fn(model, x_start_residual, target_layer, component_y_input, reset_hooks_end=False, start_at_layer=0, tokens=y_tokens)
    answer_y_logit = y_logits[0, -1, answer_x_idx]
    _, answer_y_idx = y_logits[0, -1].max(dim=-1)

    return answer_x_logit, answer_y_logit, answer_x_idx, answer_y_idx, x_loss, y_loss


def compute_similarity(a, b):
    cosine_similarity_loss = CosineSimilarity(dim=-1)
    return cosine_similarity_loss(a, b).mean()


def counterfactuals(start_input: str, model: HookedTransformer, target_layer: HookPoint, target_index: int, iterations=100):
    # Explicitly calculate and expose the result for each attention head
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)

    # Initialise contrastive pair
    start_tokens = model.to_tokens(start_input, prepend_bos=False)
    start_embed = model.embed(start_tokens)

    torch.set_grad_enabled(True)
    
    x = Embedding(model, start_embed)
    y = Embedding(model, start_embed)
    # Gradient descent over the raw vector values, not the projections
    contrastive_embeddings = (x, y)
    params = [x.vector, y.vector]
    optimizer = torch.optim.AdamW(params, maximize=True, lr=0.05)
    aggregator = UPGrad()

    # Discrete gradient descent
    for step in range(iterations):
        model.reset_hooks()

        x.update_projection()
        y.update_projection()

        # Maximise semantic differences between outputs 

        answer_x_logit, answer_y_logit, answer_x_idx, answer_y_idx, x_loss, y_loss = \
            compute_outputs(model, contrastive_embeddings, target_layer, target_index)

        answer_x_token = model.to_string(answer_x_idx)
        answer_y_token = model.to_string(answer_y_idx)
        print(f"Outputs: {answer_x_token} vs. {answer_y_token}")

        output_diff = (answer_x_logit - answer_y_logit).unsqueeze(0)
        print(f"Output logit diff: {output_diff}")

        answer_x_embed, answer_y_embed = model.embed(answer_x_idx), model.embed(answer_y_idx)
        output_sem_sim = compute_similarity(answer_x_embed, answer_y_embed)
        print(f"Output semantic similarity: {output_sem_sim}")

        # TODO: approximate instead of storing duplicate graph
        # output_grads = torch.autograd.grad(output_diff, [x.projection, y.projection], create_graph=True)
        # output_grads = torch.stack(list(output_grads)) # Shape [2, batch, seq_len, d_model]

        input_sim = compute_similarity(x.projection, y.projection)
        print(f"Input similarity: {input_sim}")
        
        # We want to minimise perplexity, i.e. maximise negative perplexity
        neg_ppl = -((torch.exp(x_loss) + torch.exp(y_loss)) / 2)
        print(f"Negative perplexity: {neg_ppl}")

        # losses = [output_diff, output_grads, input_similarity]
        # print(f"Losses: {output_diff.shape, output_grads.shape, input_similarity.shape}")
        # losses = [output_diff, input_sim]
        losses = [output_diff, output_sem_sim, input_sim, neg_ppl]

        optimizer.zero_grad()
        # Calculate gradients with respect to projected embeddings
        # (x.vector.grad, y.vector.grad) = torch.autograd.grad(loss, [x.projection, y.projection])
        backward(tensors=losses, aggregator=aggregator, inputs=[x.projection, y.projection])
        x.vector.grad, y.vector.grad = x.projection.grad, y.projection.grad
        # Apply gradient to continuous embeddings
        optimizer.step()

        torch.mps.empty_cache()

    x, y = contrastive_embeddings
    _, x_cf = map_projection_to_string(model, x.projection)
    _, y_cf = map_projection_to_string(model, y.projection)
    model.reset_hooks()
    return x_cf, y_cf