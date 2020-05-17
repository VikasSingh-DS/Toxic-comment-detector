import torch
import config


class DISTILBERTDataset:
    """
    We are required to give it a number of pieces of information which seem redundant, 
    or like they could easily be inferred from the data without us explicity providing it.
    This class prepare the dataset or input format to distilbert modeling.
    """

    def __init__(self, comment_text, target):
        self.comment_text = comment_text
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,  # sentence to encode
            None,
            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
            max_length=self.max_len,  # pad & truncate all sentences.
            pad_to_max_length=True,  # Map the tokens to thier word ids, mask and attention_mask
        )

        # Map the tokens to thier word ids, and mask
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        # Return pytorch tensors
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
