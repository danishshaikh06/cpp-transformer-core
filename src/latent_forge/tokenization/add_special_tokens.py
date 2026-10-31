"""Add boundary tokens to the encoded C++ training corpus."""

import json

import numpy as np

from cpp_transformer.core.paths import CONFIGS_DIR, PROCESSED_DATA_DIR


def load_special_token_ids() -> dict[str, int]:
    with (CONFIGS_DIR / "training.json").open(encoding="utf-8") as config_file:
        return json.load(config_file)["special_tokens"]


def main() -> None:
    tokens = np.load(PROCESSED_DATA_DIR / "tokens.npy")
    token_ids = load_special_token_ids()
    bos_id = token_ids["bos_id"]
    eos_id = token_ids["eos_id"]

    if int(tokens.max()) >= token_ids["pad_id"]:
        raise ValueError("Token IDs overlap with configured special-token IDs.")

    tokens_with_special = np.concatenate(([bos_id], tokens, [eos_id]))
    output_file = PROCESSED_DATA_DIR / "tokens_with_specialv2.npy"
    np.save(output_file, tokens_with_special)
    print(f"Saved {len(tokens_with_special)} tokens to {output_file}")


if __name__ == "__main__":
    main()
