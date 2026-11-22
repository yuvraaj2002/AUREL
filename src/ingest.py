"""Build a local embedding index from data/art/*/meta.json.

Walks each artwork, joins interpretations into one embed_text, embeds with
Qwen3-Embedding-0.6B, then writes:

    data/embeddings/vectors.npy      # (N, 1024) float16
    data/embeddings/metadata.json   # length N, row i ↔ vectors[i]

Examples:
    source .venv/bin/activate
    python src/ingest.py
    python src/ingest.py --art-dir data/art --out-dir data/embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from embed import EMBED_DIM, MODEL_ID, QwenEmbedder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ART_DIR = ROOT / "data" / "art"
DEFAULT_OUT_DIR = ROOT / "data" / "embeddings"

FACET_LABELS = {
    "philosophical_reading": "Philosophical",
    "emotional_tone": "Emotional",
    "visual_notes": "Visual",
}

# Lean index row — full corpus lives in data/art/*/meta.json
_META_KEYS = ("art_id", "title", "artist", "theme")


def build_embed_text(meta: dict) -> str:
    """Join interpretations into one passage for embedding."""
    interpretations = meta.get("interpretations") or []
    if not interpretations:
        raise ValueError(f"{meta.get('art_id')}: no interpretations")

    lines = [
        f"{meta['title']} — {meta['artist']} ({meta['year']}).",
        f"Theme: {meta['theme']}.",
    ]
    for item in interpretations:
        label = FACET_LABELS.get(item["facet"], item["facet"].replace("_", " ").title())
        lines.append(f"{label}: {item['text'].strip()}")

    return "\n".join(lines)


def vector_metadata(meta: dict, embed_text: str) -> dict:
    row = {key: meta[key] for key in _META_KEYS}
    row["embed_text"] = embed_text
    return row


def iter_meta_files(art_dir: Path) -> list[Path]:
    files = sorted(p for p in art_dir.glob("*/meta.json") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no meta.json files under {art_dir}")
    return files


def ingest(art_dir: Path, out_dir: Path, dim: int = EMBED_DIM) -> None:
    meta_files = iter_meta_files(art_dir)
    print(f"found {len(meta_files)} artworks in {art_dir}", file=sys.stderr)
    print(f"loading {MODEL_ID} on mps …", file=sys.stderr)
    embedder = QwenEmbedder()

    vectors: list[np.ndarray] = []
    metadata: list[dict] = []

    for path in meta_files:
        meta = json.loads(path.read_text(encoding="utf-8"))
        art_id = meta.get("art_id", path.parent.name)
        embed_text = build_embed_text(meta)
        print(f"  embedding {art_id} …", file=sys.stderr)

        vec = embedder.embed(embed_text, dim=dim).numpy().astype(np.float16, copy=False)
        vectors.append(vec)
        metadata.append(vector_metadata(meta, embed_text))

    matrix = np.stack(vectors, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "vectors.npy"
    meta_path = out_dir / "metadata.json"

    np.save(vectors_path, matrix)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {vectors_path}  shape={matrix.shape} dtype={matrix.dtype}", file=sys.stderr)
    print(f"wrote {meta_path}  rows={len(metadata)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest art meta.json → local .npy index")
    parser.add_argument("--art-dir", type=Path, default=DEFAULT_ART_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dim", type=int, default=EMBED_DIM)
    args = parser.parse_args(argv)

    ingest(args.art_dir.resolve(), args.out_dir.resolve(), dim=args.dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
