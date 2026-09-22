from cpp_transformer.core.paths import PROCESSED_DATA_DIR
from cpp_transformer.tokenization.bpe_tokenizer import BPETokenizer


def main() -> None:
    with (PROCESSED_DATA_DIR / "cpp_dataset_clean.txt").open(encoding="utf-8") as dataset_file:
        text = dataset_file.read()

    tokenizer = BPETokenizer(vocab_size=3000)
    print("Training tokenizer...")
    tokenizer.train(text)
    tokenizer.save(PROCESSED_DATA_DIR / "tokenizer.json")
    print("Tokenizer saved.")


if __name__ == "__main__":
    main()
