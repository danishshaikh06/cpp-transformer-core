import argparse

import numpy as np

from cpp_transformer.core.paths import PROCESSED_DATA_DIR
from cpp_transformer.tokenization.bpe_tokenizer import BPETokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Encode the cleaned dataset into token IDs.")
    parser.add_argument(
        "--full-exact",
        action="store_true",
        help="Encode the whole dataset in one pass. This is exact but very slow with the current Python BPE implementation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Chunk size to use in practical mode.",
    )
    return parser.parse_args()


def encode_full_text(tokenizer, text):
    print("Encoding full dataset into token IDs. This may take a very long time...")
    tokens = tokenizer.encode(text)
    return np.array(tokens, dtype=np.int32)


def encode_in_chunks(tokenizer, text, chunk_size):
    print(f"Encoding dataset in chunks of {chunk_size} characters.")
    print("Note: this is much faster, but chunk boundaries can introduce small tokenization differences.")

    tokens_list = []
    total_chunks = (len(text) + chunk_size - 1) // chunk_size

    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = text[start:end]
        tokens = tokenizer.encode(chunk)
        tokens_list.append(np.array(tokens, dtype=np.int32))
        print(f"Chunk {i + 1}/{total_chunks} | tokens: {len(tokens)}")

    return np.concatenate(tokens_list)


def main():
    args = parse_args()

    tokenizer = BPETokenizer()
    tokenizer.load(PROCESSED_DATA_DIR / "tokenizer.json")
    print("Tokenizer loaded")

    with (PROCESSED_DATA_DIR / "cpp_dataset_clean.txt").open(encoding="utf-8") as f:
        text = f.read()

    print("Dataset length:", len(text))

    if args.full_exact:
        tokens_array = encode_full_text(tokenizer, text)
    else:
        tokens_array = encode_in_chunks(tokenizer, text, args.chunk_size)

    np.save(PROCESSED_DATA_DIR / "tokens.npy", tokens_array)

    print("Final token count:", len(tokens_array))
    print("Max token ID:", int(tokens_array.max()))
    print("Saved tokens.npy")


if __name__ == "__main__":
    main()
