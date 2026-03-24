#!/usr/bin/env python3
"""
Prepare pre-tokenized binary shards for autoresearch-swift.

Downloads data, trains BPE tokenizer, and converts to binary int32 files.
Self-contained — no dependency on reference repos.

Prerequisites:
    pip install tiktoken rustbpe pyarrow numpy requests

Usage:
    python3 scripts/prepare_tokens.py                 # default: 10 train shards
    python3 scripts/prepare_tokens.py --num-shards 3  # quick test
    python3 scripts/prepare_tokens.py --num-shards -1 # all shards
"""

import argparse
import os
import pickle  # required by tiktoken's serialization format
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Constants (must match reference autoresearch)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
TOKENS_DIR = os.path.join(CACHE_DIR, "tokens")

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Step 1: Download data shards
# ---------------------------------------------------------------------------

def download_single_shard(index):
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True
    url = f"{BASE_URL}/{filename}"
    for attempt in range(1, 6):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as exc:
            print(f"  Attempt {attempt}/5 failed for {filename}: {exc}")
            for p in [filepath + ".tmp", filepath]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except OSError: pass
            if attempt < 5:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, workers=8):
    os.makedirs(DATA_DIR, exist_ok=True)
    ids = list(range(min(num_shards, MAX_SHARD)))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already present")
        return
    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} cached)...")
    with Pool(processes=max(1, min(workers, needed))) as pool:
        results = pool.map(download_single_shard, ids)
    print(f"Data: {sum(results)}/{len(ids)} shards ready")


# ---------------------------------------------------------------------------
# Step 2: Train BPE tokenizer
# ---------------------------------------------------------------------------

def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in paths:
        pf = pq.ParquetFile(filepath)
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg).column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    import rustbpe
    import tiktoken

    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    if len(list_parquet_files()) < 2:
        print("Need at least 2 shards. Download more data first.")
        sys.exit(1)

    print("Tokenizer: training BPE...")
    t0 = time.time()
    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(text_iterator(), VOCAB_SIZE - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)

    pattern = tok.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    offset = len(mergeable_ranks)
    special = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="rustbpe", pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special)

    # tiktoken uses pickle for serialization — this is their standard format
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer: trained in {time.time() - t0:.1f}s")

    # Build token_bytes lookup for BPB evaluation
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for tid in range(enc.n_vocab):
        s = enc.decode([tid])
        token_bytes_list.append(0 if s in special_set else len(s.encode("utf-8")))
    np.save(token_bytes_path, np.array(token_bytes_list, dtype=np.int32))
    print(f"Tokenizer: vocab_size={enc.n_vocab}")


# ---------------------------------------------------------------------------
# Step 3: Tokenize to binary shards
# ---------------------------------------------------------------------------

def load_tokenizer():
    import tiktoken  # noqa: F811
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    # Loading our own trained tokenizer (not untrusted data)
    with open(tokenizer_pkl, "rb") as f:
        enc = pickle.load(f)  # noqa: S301
    bos_id = enc.encode_single_token(BOS_TOKEN)
    return enc, bos_id


def tokenize_to_binary():
    os.makedirs(TOKENS_DIR, exist_ok=True)
    enc, bos_id = load_tokenizer()
    parquet_paths = list_parquet_files()
    total_tokens = 0

    for parquet_path in parquet_paths:
        shard_name = os.path.splitext(os.path.basename(parquet_path))[0]
        bin_path = os.path.join(TOKENS_DIR, f"{shard_name}.bin")
        if os.path.exists(bin_path):
            total_tokens += os.path.getsize(bin_path) // 4
            print(f"  {shard_name}.bin exists, skipping")
            continue

        print(f"  Tokenizing {shard_name}...", end=" ", flush=True)
        t0 = time.time()
        pf = pq.ParquetFile(parquet_path)
        all_tokens = []
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg).column("text").to_pylist():
                ids = enc.encode_ordinary(text)
                all_tokens.append(bos_id)
                all_tokens.extend(ids)

        with open(bin_path, "wb") as f:
            f.write(struct.pack(f"<{len(all_tokens)}i", *all_tokens))
        total_tokens += len(all_tokens)
        print(f"{len(all_tokens):,} tokens ({time.time() - t0:.1f}s)")

    print(f"\nTotal: {total_tokens:,} tokens in {TOKENS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch-swift")
    parser.add_argument("--num-shards", type=int, default=10, help="Training shards (-1 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Download workers")
    args = parser.parse_args()
    num = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache: {CACHE_DIR}\n")
    print("=== Step 1: Download ===")
    download_data(num, args.workers)
    print("\n=== Step 2: Tokenizer ===")
    train_tokenizer()
    print("\n=== Step 3: Binary shards ===")
    tokenize_to_binary()
    print("\nDone! Run: swift build -c release && .build/release/AutoResearch")


if __name__ == "__main__":
    main()
