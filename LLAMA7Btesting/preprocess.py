from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_data(dataset_name="ag_news"):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, split="test[:50]")

    def tokenize_fn(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"  # Ensure PyTorch tensors
        )
        example["input_ids"] = tokens["input_ids"]
        example["attention_mask"] = tokens["attention_mask"]
        return example

    return dataset.map(tokenize_fn)
