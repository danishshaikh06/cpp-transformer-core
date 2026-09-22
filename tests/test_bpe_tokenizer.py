from cpp_transformer.tokenization.bpe_tokenizer import BPETokenizer


def test_bpe_tokenizer_round_trip() -> None:
    tokenizer = BPETokenizer(vocab_size=260)
    text = "int main() { return 0; }"
    tokenizer.train(text)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_tokenizer_persists_merges(tmp_path) -> None:
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("abababab")
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_file)

    restored = BPETokenizer()
    restored.load(tokenizer_file)

    assert restored.encode("abab") == tokenizer.encode("abab")
