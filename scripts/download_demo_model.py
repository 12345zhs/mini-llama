#!/usr/bin/env python3
"""
从魔塔社区下载一个Qwen2-0.5B的GGUF量化版
"""

import argparse
import os
import sys
import urllib.request

DEFAULT_REPO = "qwen/Qwen2-0.5B-Instruct-GGUF"
DEFAULT_REMOTE_FILE = "qwen2-0_5b-instruct-q8_0.gguf"
DEFAULT_FILE = "Qwen2-0.5B-Instruct-Q8_0.gguf"
DEFAULT_DIR = "models/chat"


def is_valid_gguf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def get_modelscope_download_url(repo: str, filename: str) -> str:
    return f"https://modelscope.cn/models/{repo}/resolve/master/{filename}"


def download_file(url: str, dest: str):
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    last_pct = -1

    def reporthook(block_num, block_size, total_size):
        nonlocal last_pct
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            pct_int = int(pct)
            if pct_int == last_pct and pct_int < 100:
                return
            last_pct = pct_int
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline after progress


def main():
    parser = argparse.ArgumentParser(description="Download demo GGUF chat model")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="ModelScope repo")
    parser.add_argument(
        "--remote-file",
        default=DEFAULT_REMOTE_FILE,
        help="GGUF filename in the remote ModelScope repo",
    )
    parser.add_argument("--file", default=DEFAULT_FILE, help="Local GGUF filename")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="Local download directory")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    dest = os.path.join(args.dir, args.file)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        if is_valid_gguf(dest):
            print(f"Model already exists: {dest} ({size_mb:.1f} MB)")
            return 0
        reason = "Git LFS pointer" if is_git_lfs_pointer(dest) else "invalid GGUF file"
        print(f"Replacing {reason}: {dest} ({size_mb:.1f} MB)")
        os.remove(dest)

    url = get_modelscope_download_url(args.repo, args.remote_file)
    try:
        download_file(url, dest)
    except Exception as e:
        print(f"Error downloading: {e}", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    if not is_valid_gguf(dest):
        print(f"Downloaded file is not a valid GGUF: {dest}", file=sys.stderr)
        return 1
    print(f"Downloaded: {dest} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
