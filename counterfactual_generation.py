import torch
import numpy as np

from transformer_lens.utils import get_act_name, get_device
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from sentence_transformers.util import semantic_search, normalize_embeddings, dot_score


class Embedding:

    def __init__(self, d_embed, project_fn, vector):
        self.d_embed = d_embed
        self.project_fn = project_fn
        # Continuous-valued embedding vector
        self.vector = vector
        # Projected embedding vector (maps to model tokens)
        self.projection = project_fn(vector)

    def update(self, new_vector):
        self.vector = new_vector
        self.projection = self.project_fn(new_vector)


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
        curr_embeds = curr_embeds.reshape((-1, d_model))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        norm_embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, norm_embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        projected_embeds = torch.gather(input=embedding_matrix, dim=0, index=nn_indices)

        projected_embeds = projected_embeds.reshape((batch_size, seq_len, d_model))

    return projected_embeds, nn_indices


def counterfactuals(baseline: str, model: HookedTransformer, target_component: HookPoint):
    # Initialise contrastive pair
    x, y = Embedding(model.cfg.d_model, )