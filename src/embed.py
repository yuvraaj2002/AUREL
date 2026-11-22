"""Minimal text embedding with Qwen3-Embedding-0.6B (MPS).

Examples:
    source .venv/bin/activate
    python src/embed.py "wisdom through contemplation"
    python src/embed.py "stillness before the storm" --dim 512
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024
DEVICE = torch.device("mps")


def _last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), seq_lens]


class QwenEmbedder:
    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16
        ).to(DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, text: str, dim: int = EMBED_DIM) -> torch.Tensor:
        """Embed one string → 1-D float16 tensor of shape (dim,), L2-normalized."""
        batch = self.tokenizer(
            [text], padding=True, truncation=True, max_length=8192, return_tensors="pt"
        ).to(DEVICE)

        out = self.model(**batch)
        pooled = _last_token_pool(out.last_hidden_state, batch["attention_mask"])
        if dim != EMBED_DIM:
            pooled = pooled[:, :dim]
        return F.normalize(pooled, p=2, dim=1)[0].cpu()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed a string with Qwen3-Embedding-0.6B")
    parser.add_argument("text", help="Text to embed")
    parser.add_argument("--dim", type=int, default=EMBED_DIM)
    args = parser.parse_args(argv)

    print(f"loading {MODEL_ID} on mps …", file=sys.stderr)
    vec = QwenEmbedder().embed(args.text, dim=args.dim)
    print(f"shape={tuple(vec.shape)} dtype={vec.dtype}")
    print(vec.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
