#!/usr/bin/env python3
"""
Prepare pre-tokenized binary shards for autoresearch-swift.

This script:
1. Downloads data shards from HuggingFace (if not already present)
2. Trains a BPE tokenizer (if not already trained)
3. Tokenizes all parquet shards and writes binary int32 files to ~/.cache/autoresearch/tokens/
4. Creates token_bytes.npy for evaluation

Prerequisites:
    pip install tiktoken rustbpe pyarrow numpy requests

Usage:
    python3 scripts/prepare_tokens.py                 # default: 10 train shards
    python3 scripts/prepare_tokens.py --num-shards 3  # quick test: 3 shards
    python3 scripts/prepare_tokens.py --num-shards -1 # all shards
"""

import argparse
import os
import struct
import sys
import time

# Add reference directory to path for shared code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REFERENCE_DIR = os.path.join(PROJECT_ROOT, "references", "autoresearch-mlx")

# Import prepare module from reference
sys.path.insert(0, REFERENCE_DIR)
try:
    import prepare as ref_prepare
except ImportError:
    print("ERROR: Cannot import reference prepare.py from", REFERENCE_DIR)
    print("Make sure the reference repos are cloned:")
    print("  git clone https://github.com/trevin-creator/autoresearch-mlx references/autoresearch-mlx")
    sys.exit(1)

# Our additional output directory
TOKENS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokens")


def tokenize_shards_to_binary():
    """Convert parquet shards to binary int32 token files for Swift consumption."""
    os.makedirs(TOKENS_DIR, exist_ok=True)

    tokenizer = ref_prepare.Tokenizer.from_directory()
    bos_token = tokenizer.get_bos_token_id()
    parquet_paths = ref_prepare.list_parquet_files()

    if not parquet_paths:
        print("No parquet files found. Run download first.")
        return

    import pyarrow.parquet as pq

    total_tokens = 0
    for parquet_path in parquet_paths:
        shard_name = os.path.splitext(os.path.basename(parquet_path))[0]
        bin_path = os.path.join(TOKENS_DIR, f"{shard_name}.bin")

        if os.path.exists(bin_path):
            # Count existing tokens
            total_tokens += os.path.getsize(bin_path) // 4
            print(f"  {shard_name}.bin already exists, skipping")
            continue

        print(f"  Tokenizing {shard_name}...", end=" ", flush=True)
        t0 = time.time()

        parquet_file = pq.ParquetFile(parquet_path)
        all_tokens = []

        for rg_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(rg_idx)
            texts = row_group.column("text").to_pylist()

            for text in texts:
                doc_tokens = tokenizer.encode(text, prepend=bos_token)
                if isinstance(doc_tokens, list) and len(doc_tokens) > 0:
                    if isinstance(doc_tokens[0], list):
                        # batch result
                        for t in doc_tokens:
                            all_tokens.extend(t)
                    else:
                        all_tokens.extend(doc_tokens)

        # Write as raw int32 binary
        with open(bin_path, "wb") as f:
            f.write(struct.pack(f"<{len(all_tokens)}i", *all_tokens))

        dt = time.time() - t0
        total_tokens += len(all_tokens)
        print(f"{len(all_tokens):,} tokens ({dt:.1f}s)")

    print(f"\nTotal: {total_tokens:,} tokens across {len(parquet_paths)} shards")
    print(f"Binary tokens saved to: {TOKENS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Prepare binary token shards for autoresearch-swift")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Number of training shards to download (-1 = all)")
    parser.add_argument("--download-workers", type=int, default=8,
                        help="Parallel download workers")
    args = parser.parse_args()

    num_shards = ref_prepare.MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {ref_prepare.CACHE_DIR}")
    print(f"Tokens directory: {TOKENS_DIR}")
    print()

    # Step 1: Download parquet shards
    print("=== Step 1: Download data shards ===")
    ref_prepare.download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    print("=== Step 2: Train BPE tokenizer ===")
    ref_prepare.train_tokenizer()
    print()

    # Step 3: Convert to binary token shards
    print("=== Step 3: Tokenize → binary shards ===")
    tokenize_shards_to_binary()
    print()

    print("Done! Ready to train with autoresearch-swift.")
    print(f"  swift build -c release && .build/release/AutoResearch")


if __name__ == "__main__":
    main()
