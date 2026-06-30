from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TOKENIZER_DIR = ROOT / "tokenizer" / "llama-2-7b-hf"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
MODELSCOPE_MODEL_ID = "shakechen/Llama-2-7b-hf"


def download_tiny_shakespeare(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "tinyshakespeare.txt"
    if out.exists() and not force:
        print(f"Using existing dataset: {out}")
        return out

    print(f"Downloading TinyShakespeare text to {out}")
    with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=60) as response:
        text = response.read()
    out.write_bytes(text)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return out


def download_modelscope_tokenizer(force: bool = False) -> Path:
    if TOKENIZER_DIR.exists() and any(TOKENIZER_DIR.iterdir()) and not force:
        print(f"Using existing tokenizer directory: {TOKENIZER_DIR}")
        return TOKENIZER_DIR

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:  # pragma: no cover - depends on optional install state
        raise RuntimeError(
            "modelscope is required to download the tokenizer. Install requirements first."
        ) from exc

    allow_patterns = [
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    print(f"Downloading tokenizer files from ModelScope model {MODELSCOPE_MODEL_ID}")
    snapshot_path = Path(
        snapshot_download(
            MODELSCOPE_MODEL_ID,
            local_dir=str(TOKENIZER_DIR),
            allow_file_pattern=allow_patterns,
        )
    )

    if snapshot_path.resolve() != TOKENIZER_DIR.resolve() and snapshot_path.exists():
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        for pattern in allow_patterns:
            for src in snapshot_path.glob(pattern):
                shutil.copy2(src, TOKENIZER_DIR / src.name)

    print(f"Tokenizer available at {TOKENIZER_DIR}")
    return TOKENIZER_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Download assignment data and tokenizer assets.")
    parser.add_argument("--force", action="store_true", help="Re-download assets when present.")
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Only download TinyShakespeare text.",
    )
    args = parser.parse_args()

    download_tiny_shakespeare(force=args.force)
    if not args.skip_tokenizer:
        download_modelscope_tokenizer(force=args.force)

    print("Asset preparation finished.")


if __name__ == "__main__":
    os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / ".modelscope-cache"))
    main()
