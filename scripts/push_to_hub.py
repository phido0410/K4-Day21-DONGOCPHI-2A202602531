#!/usr/bin/env python3
"""Bonus B5: Push adapter to Hugging Face Hub for Lab 21.

Usage:
    python scripts/push_to_hub.py --repo-id <username>/lab21-qwen35-triage-vi [--token hf_xxx]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Vui lòng cài đặt huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Push fine-tuned adapter to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo ID, ví dụ: your-username/lab21-qwen35-triage-vi")
    parser.add_argument("--adapter-dir", type=str, default=str(ROOT / "adapters" / "correct"), help="Đường dẫn thư mục adapter")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (hoặc đặt biến môi trường HF_TOKEN)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    adapter_path = pathlib.Path(args.adapter_dir)
    if not (adapter_path / "adapter_model.safetensors").exists():
        print(f"Lỗi: Không tìm thấy adapter tại {adapter_path}. Hãy chạy NB3 trước!")
        sys.exit(1)

    print(f"Đang đẩy adapter từ {adapter_path} lên https://huggingface.co/{args.repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload Lab 21 Qwen3.5-4B LoRA Vietnamese Customer Support Triage Adapter",
    )

    url = f"https://huggingface.co/{args.repo_id}"
    print(f"Thành công! Adapter công khai tại: {url}")

    # Ghi nhận vào LINKS.md
    links_file = ROOT / "LINKS.md"
    content = f"# Lab 21 — Artifact Links\n\n- Hugging Face Model: [{args.repo_id}]({url})\n"
    links_file.write_text(content, encoding="utf-8")
    print(f"Đã cập nhật link vào LINKS.md")


if __name__ == "__main__":
    main()

