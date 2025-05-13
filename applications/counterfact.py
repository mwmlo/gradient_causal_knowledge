import pandas as pd
import re
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

        # Default to first corrupt candidate
        corrupt_prompt = row["attribute_prompts"][0]

        # Choose the corrupt prompt which has the same sentence structure
        # Convert template to regex pattern (escape and replace {})
        template_regex = re.escape(template).replace(r"\{\}", r".+")
        pattern = re.compile(f"^{template_regex}$")

        for corrupt_candidate in row["attribute_prompts"]:
            if bool(pattern.fullmatch(corrupt_candidate)):
                corrupt_prompt = corrupt_candidate

        # Label includes index of original (clean) token, and index of rewritten (corrupt) token
        original_token = row["requested_rewrite.target_true.str"]
        original_idx = self.model.to_single_token(original_token)

        rewritten_token = row["requested_rewrite.target_new.str"]
        rewritten_idx = self.model.to_single_token(rewritten_token)

        label = [original_idx, rewritten_idx]

        return original_prompt, corrupt_prompt, label

    def to_dataloader(self, batch_size: int):

        def collate(xs):
            clean, corrupted, labels = zip(*xs)
            clean = list(clean)
            corrupted = list(corrupted)
            return clean, corrupted, labels

        return DataLoader(self, batch_size=batch_size, collate_fn=collate)
