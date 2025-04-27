from datasets import load_dataset

try:
    dataset = load_dataset("wikiann", "vi", trust_remote_code=True)
    print("Splits:", dataset.keys())
    print("Train features:", dataset["train"].features)
    print("Sample train data:", dataset["train"][0])
except Exception as e:
    print("Lỗi:", e)