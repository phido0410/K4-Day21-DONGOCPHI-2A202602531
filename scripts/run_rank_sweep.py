#!/usr/bin/env python3
"""Bonus B4: Controlled rank sweep for Lab 21.

Fix target_modules="text-linear", LR=1e-4, same max_steps as NB3's `correct`.
Sweeps r in {8, 64} (r=16 is `correct`).
Evaluates all three on the target set to answer:
  "Is rank the lever, or are placement and LR far larger levers?"
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from labkit import data, evaluate as ev, generate, modeling, report, train
from labkit.config import LoraSpec, SPECS, get_tier, training_epochs


def run_rank_sweep():
    tier = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
    print(f"=== Bonus B4: Controlled Rank Sweep on {tier.name} ===")

    split_dir = ROOT / "data" / "split"
    assert split_dir.exists(), "Chạy NB1 trước để có data/split/"
    train_rows = [json.loads(line) for line in open(split_dir / "train.jsonl", encoding="utf-8") if line.strip()]

    target_data = [json.loads(line) for line in open(ROOT / "data" / "eval_target.jsonl", encoding="utf-8") if line.strip()]
    eval_limit = int(os.environ.get("EVAL_LIMIT", "0"))
    if eval_limit:
        target_data = target_data[:eval_limit]

    epochs = training_epochs()
    max_steps = train.planned_steps(len(train_rows), tier, epochs)
    print(f"Step budget: {max_steps} steps (cùng số step với NB3/NB4)")

    # Base configuration template from `correct` (r=16)
    base_spec = SPECS["correct"]

    sweep_ranks = [8, 64]
    results = []

    # Check if correct already exists in results/autopsy.json or runs.csv
    runs = {r.get("run"): r for r in report.read_rows("runs.csv", results_dir=ROOT / "results")}
    correct_row = runs.get("correct")

    # Load baseline correct score from autopsy.json if available
    autopsy_path = ROOT / "results" / "autopsy.json"
    correct_target_score = None
    if autopsy_path.exists():
        autopsy_data = json.loads(autopsy_path.read_text(encoding="utf-8"))
        for item in autopsy_data:
            if item.get("run") == "correct":
                correct_target_score = item.get("target")

    results.append({
        "run": "correct",
        "r": 16,
        "alpha": 32,
        "trainable_params": int(correct_row["trainable_params"]) if correct_row else 32464896,
        "final_loss": float(correct_row["final_loss"]) if correct_row else None,
        "target": correct_target_score,
        "train_seconds": float(correct_row["train_seconds"]) if correct_row else None,
        "peak_vram_gb": float(correct_row["peak_vram_gb"]) if correct_row else None,
    })

    train_ds = None

    for r in sweep_ranks:
        key = f"rank_{r}"
        spec = LoraSpec(
            key=key,
            r=r,
            alpha=2 * r,
            target="text-linear",
            lr=base_spec.lr,
            load_in_4bit=False,
            label=f"text-linear · r={r} · alpha={2*r}",
            teaches=f"Bonus B4: Rank sweep r={r}",
        )

        adapter_dir = ROOT / "adapters" / key
        model_exists = (adapter_dir / "adapter_model.safetensors").exists()
        force = os.environ.get("FORCE_RETRAIN", "").lower() in {"1", "true", "yes"}

        if not model_exists or force:
            print(f"\n--- Training {key} (r={r}, alpha={2*r}) ---")
            model, tok = generate.load_base(tier)
            if train_ds is None:
                train_ds = Dataset.from_list(
                    data.to_training_dataset(
                        tok,
                        train_rows,
                        max_length=tier.max_length,
                        mask_mode=os.environ.get("MASK_MODE", "assistant-only"),
                    )
                )

            targets = modeling.resolve_target_modules(model, spec.target)
            trainable = modeling.count_lora_params(model, targets, spec.r)

            want = train.sft_config_kwargs(tier, spec, str(adapter_dir), max_steps=max_steps)
            sft_kwargs, _ = train.filter_kwargs(SFTConfig, want, label=f"SFTConfig[{key}]")
            lora_kwargs, _ = train.filter_kwargs(
                LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[{key}]"
            )

            trainer = SFTTrainer(
                model=model,
                args=SFTConfig(**sft_kwargs),
                train_dataset=train_ds,
                processing_class=tok,
                peft_config=LoraConfig(**lora_kwargs),
            )
            train.align_trainable_precision(trainer.model)

            t0 = time.perf_counter()
            res = trainer.train()
            elapsed = time.perf_counter() - t0

            trainer.model.save_pretrained(adapter_dir)
            tok.save_pretrained(adapter_dir)

            peak_vram = generate.peak_vram_gb()
            final_loss = round(res.training_loss, 4)

            del trainer, model
            generate.free_memory()
        else:
            print(f"\nAdapter {key} already exists at {adapter_dir}, skipping training.")
            final_loss = None
            elapsed = None
            peak_vram = None
            trainable = None

        # Evaluate on target set
        print(f"Scoring {key} on target set ({len(target_data)} items)...")
        model, tok = generate.load_base(tier)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

        preds, lat = generate.generate_batch(
            model, tok, [item["input"] for item in target_data],
            system=generate.NAIVE_PROMPT,
            label=f"{key}/target"
        )
        tgt = sum(ev.triage_field_accuracy(p, item["label"]) for p, item in zip(preds, target_data)) / len(target_data)
        fmt = sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds) / len(preds)

        del model
        generate.free_memory()

        print(f"[{key}] target: {tgt:.4f} | format: {fmt:.4f} | latency: {lat:.1f}ms")

        results.append({
            "run": key,
            "r": r,
            "alpha": 2 * r,
            "trainable_params": trainable,
            "final_loss": final_loss,
            "target": round(tgt, 4),
            "format": round(fmt, 4),
            "latency_ms": round(lat, 1),
            "train_seconds": round(elapsed, 1) if elapsed else None,
            "peak_vram_gb": peak_vram,
        })

    # Sort by rank
    results.sort(key=lambda x: x["r"])

    print("\n" + "=" * 70)
    print("BONUS B4 — KẾT QUẢ QUÉT RANK (CONTROLLED RANK SWEEP):")
    print(report.markdown_table(results, ["run", "r", "alpha", "target", "format", "latency_ms", "final_loss"]))
    print("=" * 70)

    report.write_json(results, "rank_sweep.json", results_dir=ROOT / "results")
    print(f"Đã lưu kết quả vào results/rank_sweep.json")


if __name__ == "__main__":
    run_rank_sweep()

