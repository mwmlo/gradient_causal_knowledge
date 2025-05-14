from transformer_lens import HookedTransformer
from torch import Tensor
import torch
import ast
from applications.datasets import CounterFact, CounterFactEvaluation


def efficacy_scores(
    model: HookedTransformer, prompts: list[str], answer_labels: Tensor, verbose=False
):
    n_samples = len(prompts)
    original_label = answer_labels[:, 0]
    target_label = answer_labels[:, 1]
    logits = model.forward(prompts)[:, -1, :]
    logit_probs = torch.softmax(logits, dim=-1)

    if verbose:
        print(f"Prompts: {prompts}")
        print(f"Original label: {model.to_string(original_label)}")
        print(f"Target label: {model.to_string(target_label)}")
        outputs = torch.argmax(logit_probs, dim=-1)
        print(f"Outputs: {[model.to_string(o) for o in outputs]}")

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


def evaluate_counterfact_efficacy(model: HookedTransformer, prompt_index: int, verbose=False):
    # Evaluate prompt for the same subject and relation as the original prompt, phrased similarly
    counterfact_dataset = CounterFactEvaluation(model, "generation_prompts")
    original_input, labels = counterfact_dataset.get_single_sample(prompt_index)
    model.eval()
    return efficacy_scores(model, original_input, labels, verbose=verbose)


def evaluate_counterfact_paraphrased(model: HookedTransformer, prompt_index: int, verbose=False):
    # Evaluate prompts for the same subject and relation as the original prompt, rephrased with extra content

    paraphrased_dataset = CounterFactEvaluation(model, "paraphrase_prompts")
    paraphrased_inputs, labels = paraphrased_dataset.get_single_sample(prompt_index)
    model.eval()
    return efficacy_scores(model, paraphrased_inputs, labels, verbose=verbose)


def evaluate_counterfact_neighborhood(model: HookedTransformer, prompt_index: int, verbose=False):
    # Evaluate prompts for a similar subject, relation and object as the original prompt

    neighborhood_dataset = CounterFactEvaluation(model, "neighborhood_prompts")
    neighborhood_inputs, labels = neighborhood_dataset.get_single_sample(prompt_index)
    model.eval()
    return efficacy_scores(model, neighborhood_inputs, labels, verbose=verbose)
