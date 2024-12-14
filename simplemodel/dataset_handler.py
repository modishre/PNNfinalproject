from datasets import load_dataset
from transformers import AutoTokenizer


def load_ag_news(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is valid

    dataset = load_dataset("ag_news")
    max_length = 64

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return tokenized_datasets
