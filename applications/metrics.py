from transformer_lens import HookedTransformer
from torch import Tensor
import torch
import ast
from applications.datasets import CounterFact


def efficacy_scores(
    model: HookedTransformer, prompts: list[str], answer_labels: Tensor
):
    n_samples = len(prompts)
    original_label = answer_labels[:, 0]
    target_label = answer_labels[:, 1]
    logits = model.forward(prompts)
    logit_probs = torch.softmax(logits, dim=-1)

    # Compute fraction of sample where new facts are more likely than original facts
    count_edited = torch.count_nonzero(
        logit_probs[:, target_label] > logit_probs[:, original_label]
    )
    efficacy_score = count_edited / n_samples

    # Compute mean difference between probabilities of new facts and original facts
    efficacy_magnitude = (
        logit_probs[:, target_label] - logit_probs[:, original_label]
    ).mean()

    return efficacy_score, efficacy_magnitude


def evaluate_counterfact_efficacy(model: HookedTransformer, prompt_index: int):
    # Evaluate prompt for the same subject and relation as the original prompt, phrased similarly
    counterfact_dataset = CounterFact(model)
    original_input, _, labels = counterfact_dataset.get_single_sample(prompt_index)
    return efficacy_scores(model, original_input, labels)


def evaluate_counterfact_paraphrased(model: HookedTransformer, prompt_index: int):
    # Evaluate prompts for the same subject and relation as the original prompt, rephrased with extra content

    class CounterFactParaphrased(CounterFact):

        def __init__(self, model: HookedTransformer):
            super().__init__(model)

        def __getitem__(self, index):
            # Override retrieval of prompts and answers to get paraphrased content
            row = self.df.iloc[index]

            # List of paraphrased prompts
            paraphrased_prompts = ast.literal_eval(row["paraphrase_prompts"])

            # Label includes index of original (clean) token, and index of rewritten (corrupt) token
            original_token = row["requested_rewrite.target_true.str"]
            original_idx = self.model.to_tokens(original_token, prepend_bos=False)[
                :, 0
            ].item()

            rewritten_token = row["requested_rewrite.target_new.str"]
            rewritten_idx = self.model.to_tokens(rewritten_token, prepend_bos=False)[
                :, 0
            ].item()

            label = [original_idx, rewritten_idx]

            return paraphrased_prompts, None, label

    paraphrased_dataset = CounterFactParaphrased(model)
    paraphrased_inputs, labels = paraphrased_dataset.get_single_sample(prompt_index)
    return efficacy_scores(model, paraphrased_inputs, labels)


def evaluate_counterfact_specificity(model: HookedTransformer, prompt_index: int):
    # Evaluate prompts for a similar subject, relation and object as the original prompt

    class CounterFactNeighborhood(CounterFact):

        def __init__(self, model: HookedTransformer):
            super().__init__(model)

        def __getitem__(self, index):
            # Override retrieval of prompts and answers to get paraphrased content
            row = self.df.iloc[index]

            # List of paraphrased prompts
            neighborhood_prompts = ast.literal_eval(row["neighborhood_prompts"])

            # Label includes index of original (clean) token, and index of rewritten (corrupt) token
            original_token = row["requested_rewrite.target_true.str"]
            original_idx = self.model.to_tokens(original_token, prepend_bos=False)[
                :, 0
            ].item()

            rewritten_token = row["requested_rewrite.target_new.str"]
            rewritten_idx = self.model.to_tokens(rewritten_token, prepend_bos=False)[
                :, 0
            ].item()

            label = [original_idx, rewritten_idx]

            return neighborhood_prompts, None, label

    neighborhood_dataset = CounterFactNeighborhood(model)
    neighborhood_inputs, labels = neighborhood_dataset.get_single_sample(prompt_index)
    return efficacy_scores(model, neighborhood_inputs, labels)
