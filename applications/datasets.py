import pandas as pd
import re
import ast
import torch
from transformer_lens import HookedTransformer
from torch.utils.data import Dataset, DataLoader


class CounterFact(Dataset):
    """
    Loads CounterFact dataset in required format.
    """

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.df = pd.read_csv(f"datasets/counterfact.csv")

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        # Returns (clean, corrupt, labels)
        row = self.df.iloc[index]

        template = row["requested_rewrite.prompt"]
        original_prompt = template.format(row["requested_rewrite.subject"])

        # Get list of corrupt candidate prompts
        corrupt_prompts = ast.literal_eval(row["attribute_prompts"])

        # Choose the corrupt prompt which has the same sentence structure
        # Convert template to regex pattern (escape and replace {})
        template_regex = re.escape(template).replace(r"\{\}", r".+")
        pattern = re.compile(f"^{template_regex}$")

        corrupt_prompt = corrupt_prompts[0]
        for corrupt_candidate in corrupt_prompts:
            if bool(pattern.fullmatch(corrupt_candidate)):
                corrupt_prompt = corrupt_candidate

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

        return original_prompt, corrupt_prompt, label

    def to_dataloader(self, batch_size: int):

        def collate(xs):
            clean, corrupted, labels = zip(*xs)
            clean = list(clean)
            corrupted = list(corrupted)
            labels = torch.tensor(labels)
            return clean, corrupted, labels

        return DataLoader(self, batch_size=batch_size, collate_fn=collate)

    def get_single_sample(self, index):
        original_prompt, corrupt_prompt, label = self.__getitem__(index)
        return [original_prompt], [corrupt_prompt], torch.tensor([label])
