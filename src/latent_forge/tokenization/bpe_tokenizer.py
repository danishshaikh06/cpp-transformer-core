import json

class BPETokenizer:

    def __init__(self, vocab_size=3000):
        self.vocab_size = vocab_size
        self.merges = {}

    # count pair frequencies
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    # merge token pairs
    def merge(self, ids, pair, idx):

        new_ids = []
        i = 0

        while i < len(ids):

            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    # train tokenizer
    def train(self, text):

        tokens = list(text.encode("utf-8"))

        num_merges = self.vocab_size - 256

        ids = list(tokens)

        for i in range(num_merges):

            stats = self.get_stats(ids)

            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i

            print(f"Merge {i}: {pair} -> {idx}")

            ids = self.merge(ids, pair, idx)

            self.merges[pair] = idx

    # encode text
    def encode(self, text):

        tokens = list(text.encode("utf-8"))

        while True:

            stats = self.get_stats(tokens)

            pair = None
            best_rank = float("inf")

            for p in stats:

                if p in self.merges and self.merges[p] < best_rank:
                    pair = p
                    best_rank = self.merges[p]

            if pair is None:
                break

            idx = self.merges[pair]

            tokens = self.merge(tokens, pair, idx)

        return tokens

    # decode tokens
    def decode(self, tokens):

        reverse_merges = {v: k for k, v in self.merges.items()}

        decoded = list(tokens)

        while True:

            expanded = False
            new_tokens = []

            for token in decoded:

                if token in reverse_merges:
                    a, b = reverse_merges[token]
                    new_tokens.extend([a, b])
                    expanded = True
                else:
                    new_tokens.append(token)

            decoded = new_tokens

            if not expanded:
                break

        return bytes(decoded).decode("utf-8", errors="replace")

    # save tokenizer
    def save(self, path):

        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}

        data = {
            "vocab_size": self.vocab_size,
            "merges": serializable_merges
        }

        with open(path, "w") as f:
            json.dump(data, f)

    # load tokenizer
    def load(self, path):

        with open(path, "r") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]

        self.merges = {}

        for k, v in data["merges"].items():
            a, b = map(int, k.split(","))
            self.merges[(a, b)] = v
