import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cpp_transformer.core.paths import (
    ARTIFACTS_DIR,
    BEST_MODEL_FILE,
    PROCESSED_DATA_DIR,
    TRAINING_CONFIG_FILE,
)
from cpp_transformer.models.decoder_only import build_transformer
from cpp_transformer.training.inference import generate_completion, load_tokenizer


DEFAULT_DATASET = PROCESSED_DATA_DIR / "fine_tuning_dataset.txt"
FINETUNE_DIR = ARTIFACTS_DIR / "finetune"
FINETUNE_CONFIG_FILE = FINETUNE_DIR / "finetune_config.json"
FINETUNE_BEST_MODEL_FILE = FINETUNE_DIR / "finetune_best_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

SAMPLE_PROMPTS = [
    "write code for hello world",
    "write a C++ function to check if a number is prime and explain it",
]


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len, stride):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        return max(((len(self.tokens) - self.seq_len - 1) // self.stride) + 1, 0)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the trained C++ model on instruction-style data.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to the formatted fine-tuning dataset.")
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_FILE), help="Base checkpoint to continue training from.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Fine-tuning learning rate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for fine-tuning.")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--stride", type=int, default=64, help="Stride between training windows.")
    parser.add_argument("--dropout", type=float, default=None, help="Optional dropout override.")
    parser.add_argument("--sample-every-epochs", type=int, default=1, help="Run sample generations every N epochs.")
    parser.add_argument("--max-train-chars", type=int, default=None, help="Optionally limit the fine-tuning text length for quick tests.")
    return parser.parse_args()


def load_base_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_lr(step, base_lr, warmup_steps=200):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


def encode_finetune_text(dataset_path: Path, tokenizer, bos_id: int, eos_id: int, max_train_chars: int | None):
    with dataset_path.open("r", encoding="utf-8") as f:
        text = f.read()

    if max_train_chars is not None and max_train_chars < len(text):
        text = text[:max_train_chars]

    token_ids = tokenizer.encode(text)
    return [bos_id] + token_ids + [eos_id], len(text)


def run_sample_generations(model, tokenizer, seq_len, pad_id, eos_id, bos_id, device):
    print("\nFine-tune sample generations:")
    for prompt in SAMPLE_PROMPTS:
        sample = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            seq_len=seq_len,
            pad_id=pad_id,
            eos_id=eos_id,
            bos_id=bos_id,
            device=device,
            max_new_tokens=160,
            temperature=0.8,
            top_k=40,
        )
        print(f"\nPrompt: {prompt}\nResponse: {sample}\n")


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    checkpoint_path = Path(args.checkpoint)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Fine-tuning dataset not found: {dataset_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")
    if not TRAINING_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Base training config not found: {TRAINING_CONFIG_FILE}")

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_base_config(TRAINING_CONFIG_FILE)
    tokenizer = load_tokenizer()

    seq_len = int(base_config["seq_len"])
    vocab_size = int(base_config["vocab_size"])
    pad_id = int(base_config["pad_id"])
    bos_id = int(base_config["bos_id"])
    eos_id = int(base_config["eos_id"])
    d_model = int(base_config["d_model"])
    layers = int(base_config["layers"])
    heads = int(base_config["heads"])
    d_ff = int(base_config["d_ff"])
    dropout = float(args.dropout if args.dropout is not None else base_config["dropout"])

    token_ids, char_count = encode_finetune_text(dataset_path, tokenizer, bos_id, eos_id, args.max_train_chars)
    total_tokens = len(token_ids)

    split = int(0.9 * total_tokens)
    train_tokens = token_ids[:split]
    val_tokens = token_ids[split:]

    train_dataset = TokenDataset(train_tokens, seq_len, args.stride)
    val_dataset = TokenDataset(val_tokens, seq_len, args.stride)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=AMP_ENABLED,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=AMP_ENABLED,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Fine-tuning dataset is too small for the current seq_len and stride.")

    model = build_transformer(
        tgt_vocab_size=vocab_size,
        tgt_seq_len=seq_len,
        d_model=d_model,
        N=layers,
        h=heads,
        dropout=dropout,
        d_ff=d_ff,
    ).to(DEVICE)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    finetune_metadata = {
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "stride": args.stride,
        "seq_len": seq_len,
        "dropout": dropout,
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "total_chars": char_count,
        "total_tokens": total_tokens,
        "device": DEVICE.type,
    }

    with FINETUNE_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(finetune_metadata, f, indent=2)

    print(f"Fine-tuning dataset: {dataset_path}")
    print(f"Characters used: {char_count}")
    print(f"Total fine-tune tokens: {total_tokens}")
    print(f"Train windows: {len(train_dataset)}")
    print(f"Validation windows: {len(val_dataset)}")
    print(f"Config saved to: {FINETUNE_CONFIG_FILE}")

    global_step = 0
    optimizer_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps_completed = 0
        pending_microbatches = 0
        progress_bar = tqdm(train_loader, desc=f"Finetune Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad()

        for x, y in progress_bar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            current_lr = get_lr(optimizer_step, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            tgt_mask = model.create_tgt_mask(x, pad_id)

            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                logits = model(x, tgt_mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            pending_microbatches += 1

            if pending_microbatches == args.accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                optimizer_step += 1
                pending_microbatches = 0

            total_loss += loss.item() * args.accum_steps
            steps_completed += 1
            tokens_seen = (global_step + 1) * args.batch_size * seq_len

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * args.accum_steps:.4f}",
                    "lr": f"{current_lr:.6f}",
                    "tokens": tokens_seen,
                }
            )

            global_step += 1

        if pending_microbatches > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            optimizer_step += 1

        avg_train_loss = total_loss / max(steps_completed, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                tgt_mask = model.create_tgt_mask(x, pad_id)
                logits = model(x, tgt_mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        perplexity = math.exp(avg_val_loss)

        print(f"\nEpoch {epoch + 1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), FINETUNE_BEST_MODEL_FILE)
            print(f"Best fine-tuned model saved to {FINETUNE_BEST_MODEL_FILE}")

        checkpoint_path = FINETUNE_DIR / f"finetune_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if (epoch + 1) % args.sample_every_epochs == 0:
            run_sample_generations(
                model=model,
                tokenizer=tokenizer,
                seq_len=seq_len,
                pad_id=pad_id,
                eos_id=eos_id,
                bos_id=bos_id,
                device=DEVICE,
            )


if __name__ == "__main__":
    main()
