import torch
from torch import Tensor
from torch.nn import CosineSimilarity
from torchjd.aggregation import MGDA
from torchjd import backward
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_device

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
        self.projection = nn_project(vector, self.embedding_matrix).detach().requires_grad_(True)
        # self.projection = self.projection.detach().clone().requires_grad_(True)


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


def run_model_with_cache(model: HookedTransformer, target_layer_name, *model_args, **model_kwargs):
    cache = {}
    
    def save_acts_hook_with_grad(act, hook):
        cache[hook.name] = act  # No detach!
        return act
    
    with model.hooks(fwd_hooks=[(target_layer_name, save_acts_hook_with_grad)]):
        logits = model.forward(*model_args, **model_kwargs)

    return logits, cache


def run_from_layer_fn(model: HookedTransformer, original_input, patch_layer, patch_input, reset_hooks_end=True, start_at_layer=None):
    """
    Run the model, patching in `patch_input` at the target layer.
    """
    def fwd_hook(act, hook):
        # Keep patch_input in the computational graph
        return patch_input + 0 * act
    
    with model.hooks(fwd_hooks=[(patch_layer.name, fwd_hook)], reset_hooks_end=reset_hooks_end):
        logits = model.forward(original_input, start_at_layer=start_at_layer)
    
    return logits


def map_projection_to_string(model: HookedTransformer, projection_embedding: Tensor):
    with torch.no_grad():
        logits = model.unembed(projection_embedding) # Shape [batch, seq_len, d_vocab]
        tokens = torch.argmax(logits, dim=-1) # Shape [batch, seq_len]
        print("Map tokens", tokens.shape)
        return model.to_string(tokens)


def compute_outputs(model: HookedTransformer, contrastive_pair: list[Embedding], target_layer, target_index):
    x, y = contrastive_pair
    target_layer_name = target_layer.name
    
    x_logits, x_cache = run_model_with_cache(model, target_layer_name, x.projection, start_at_layer=0)

    # Treat x as the "clean" input - as such, its top token is the "correct" answer 
    answer_x_logit, answer_idx = x_logits[0, -1].max(dim=-1)
    
    _, y_cache = run_model_with_cache(model, target_layer_name, y.projection, start_at_layer=0)

    component_y_input = x_cache[target_layer_name].clone()
    component_y_input[:, :, target_index] = y_cache[target_layer_name][:, :, target_index]

    y_logits = run_from_layer_fn(model, x.projection, target_layer, component_y_input, reset_hooks_end=False, start_at_layer=0)
    answer_y_logit = y_logits[0, -1, answer_idx]

    return answer_x_logit, answer_y_logit, answer_idx


def compute_similarity(contrastive_pair):
    x, y = contrastive_pair
    cosine_similarity_loss = CosineSimilarity(dim=-1)
    return cosine_similarity_loss(x.projection, y.projection)


def counterfactuals(start_input: str, model: HookedTransformer, target_layer: HookPoint, target_index: int, iterations=100):
    # Explicitly calculate and expose the result for each attention head
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)

    # Initialise contrastive pair
    start_tokens = model.to_tokens(start_input)
    start_embed = model.embed(start_tokens)

    torch.set_grad_enabled(True)
    
    x = Embedding(model, start_embed)
    y = Embedding(model, start_embed)
    # Optimise over the raw vector values, not the projections
    contrastive_embeddings = (x, y)
    optimizer = torch.optim.AdamW([x.vector, y.vector])
    aggregator = MGDA()

    # Discrete gradient descent
    for step in tqdm(range(iterations)):
        model.reset_hooks()

        answer_x_logit, answer_y_logit, answer_token = \
            compute_outputs(model, contrastive_embeddings, target_layer, target_index)

        print(f"Correct answer: {model.to_string(answer_token)}")

        output_diff = (answer_x_logit - answer_y_logit).unsqueeze(0)
        print(f"Output diff: {output_diff}")

        # TODO: approximate instead of storing duplicate graph
        # output_grads = torch.autograd.grad(output_diff, [x.projection, y.projection], create_graph=True)
        # output_grads = torch.stack(list(output_grads)) # Shape [2, batch, seq_len, d_model]

        input_similarity = compute_similarity(contrastive_embeddings)
        # TODO: calculate PPL without using losses
        # negative_perplexity = torch.Tensor([-torch.exp(x_ce_loss), -torch.exp(y_ce_loss)])

        # losses = [output_diff, output_grads, input_similarity]
        # print(f"Losses: {output_diff.shape, output_grads.shape, input_similarity.shape}")
        losses = [output_diff, input_similarity]

        optimizer.zero_grad()
        # Calculate gradients with respect to projected embeddings
        backward(tensors=-output_diff, aggregator=aggregator, inputs=[x.projection, y.projection])
        # Apply gradient to continuous embeddings
        optimizer.step()

        torch.mps.empty_cache()

    x, y = contrastive_embeddings
    x_cf = map_projection_to_string(model, x.vector)
    y_cf = map_projection_to_string(model, y.vector)
    model.reset_hooks()
    return x_cf, y_cf