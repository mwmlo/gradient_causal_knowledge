from transformer_lens import HookedTransformer
from torch import Tensor
import torch
import numpy as np
from applications.datasets import CounterFactEvaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def generate_next_token(model: HookedTransformer, prompts: list[str]):
    logits = model.forward(prompts)[:, -1, :]
    logit_probs = torch.softmax(logits, dim=-1)
    outputs = torch.argmax(logit_probs, dim=-1)
    return [model.to_string(o) for o in outputs]


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
        outputs = generate_next_token(model, prompts)
        print(f"Outputs: {outputs}")

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


def evaluate_counterfact_efficacy(
    model: HookedTransformer, prompt_index: int, verbose=False
):
    # Evaluate prompt for the same subject and relation as the original prompt, phrased similarly
    generation_dataset = CounterFactEvaluation(model, "generation_prompts")
    original_input, labels = generation_dataset.get_single_sample(prompt_index)
    model.eval()
    return efficacy_scores(model, original_input, labels, verbose=verbose)


def evaluate_counterfact_paraphrased(
    model: HookedTransformer, prompt_index: int, verbose=False
):
    # Evaluate prompts for the same subject and relation as the original prompt, rephrased with extra content

    paraphrased_dataset = CounterFactEvaluation(model, "paraphrase_prompts")
    paraphrased_inputs, labels = paraphrased_dataset.get_single_sample(prompt_index)
    model.eval()
    return efficacy_scores(model, paraphrased_inputs, labels, verbose=verbose)


def evaluate_counterfact_neighborhood(
    model: HookedTransformer, prompt_index: int, verbose=False
):
    # Evaluate prompts for a similar subject, relation and object as the original prompt

    neighborhood_dataset = CounterFactEvaluation(model, "neighborhood_prompts")
    neighborhood_inputs, labels = neighborhood_dataset.get_single_sample(prompt_index)
    model.eval()
    return efficacy_scores(model, neighborhood_inputs, labels, verbose=verbose)


def evaluate_consistency(model: HookedTransformer, prompt_index: int, verbose=False):
    # Generate text starting with original prompts
    generation_dataset = CounterFactEvaluation(model, "generation_prompts")
    generation_prompts, _ = generation_dataset.get_single_sample(prompt_index)
    generation_outputs = model.generate(
        generation_prompts, max_new_tokens=5, do_sample=False
    )
    generation_answers = [
        output.removeprefix(prompt)
        for output, prompt in zip(generation_outputs, generation_prompts)
    ]

    # Generate text starting with reference prompts with rewritten target attributes
    reference_dataset = CounterFactEvaluation(model, "attribute_prompts")
    reference_prompts, labels = reference_dataset.get_single_sample(prompt_index)
    reference_outputs = model.generate(
        reference_prompts, max_new_tokens=5, do_sample=False
    )
    reference_answers = [
        output.removeprefix(prompt)
        for output, prompt in zip(reference_outputs, reference_prompts)
    ]

    if verbose:
        print("Generated answers:", generation_answers)
        print("Reference answers:", reference_answers)

    # Compute cosine similarity of TF-IDF unigram vectors
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(generation_answers + reference_answers)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Remove diagonal, where self-similarity = 1
    np.fill_diagonal(similarity_matrix, np.nan)
    average_similarity = np.nanmean(similarity_matrix)

    return average_similarity
