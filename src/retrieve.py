"""Semantic retriever over the local .npy art index.

Embeds the user query with Qwen3-Embedding-0.6B, scores cosine similarity
against data/embeddings/vectors.npy, and returns the top-k artworks.

Examples:
    source .venv/bin/activate
    python src/retrieve.py "solitude facing the infinite sea"
    python src/retrieve.py "stoic courage before death" --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from embed import MODEL_ID, QwenEmbedder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = ROOT / "data" / "embeddings"

# Qwen3-Embedding is instruction-aware: queries get a task prefix; docs do not.
QUERY_INSTRUCTION = (
    "Given a user's philosophical idea, mood, or theme, retrieve artworks "
    "whose interpretations best match the intent"
)


def format_query(text: str, instruction: str = QUERY_INSTRUCTION) -> str:
    return f"Instruct: {instruction}\nQuery:{text}"


class ArtRetriever:
    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR, embedder: QwenEmbedder | None = None):
        vectors_path = index_dir / "vectors.npy"
        meta_path = index_dir / "metadata.json"
        if not vectors_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"missing index under {index_dir} — run: python src/ingest.py"
            )

        self.vectors = np.load(vectors_path).astype(np.float32)  # (N, D)
        self.metadata: list[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
        if len(self.metadata) != len(self.vectors):
            raise ValueError(
                f"index mismatch: {len(self.vectors)} vectors vs {len(self.metadata)} metadata rows"
            )

        print(f"loading {MODEL_ID} on mps …", file=sys.stderr)
        self.embedder = embedder or QwenEmbedder()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        q = self.embedder.embed(format_query(query.strip()), dim=self.vectors.shape[1])
        q = q.numpy().astype(np.float32)
        # vectors + query are L2-normalized → cosine == dot product
        scores = self.vectors @ q

        k = min(top_k, len(scores))
        top_idx = np.argsort(scores)[::-1][:k]

        results: list[dict] = []
        for rank, idx in enumerate(top_idx, start=1):
            row = dict(self.metadata[int(idx)])
            row["score"] = float(scores[int(idx)])
            row["rank"] = rank
            results.append(row)
        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Top-k semantic art retrieval")
    parser.add_argument("query", help="User idea / mood / theme")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--json", action="store_true", help="Print full JSON results")
    args = parser.parse_args(argv)

    retriever = ArtRetriever(index_dir=args.index_dir.resolve())
    hits = retriever.search(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0

    print(f"\nquery: {args.query!r}\ntop {len(hits)}:\n")
    for hit in hits:
        print(f"{hit['rank']}. {hit['title']}  ({hit['art_id']})")
        print(f"   artist: {hit['artist']}")
        print(f"   theme:  {hit['theme']}")
        print(f"   score:  {hit['score']:.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
