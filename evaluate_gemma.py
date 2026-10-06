"""Measure Gemma 4 accuracy on a held-out subset of competition_math."""

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path

from math_verify import parse, verify
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from data_loader import MathDatasetLoader


MODEL_ID = "google/gemma-4-E2B-it"
SYSTEM_PROMPT = (
    "Solve the mathematics problem step by step. Put only your final answer "
    "inside \\boxed{} using LaTeX."
)


def extract_boxed(text):
    """Return the contents of the last balanced ``\\boxed{...}`` expression."""
    marker = "\\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        content_start = start + len(marker)
        depth = 1
        for index in range(content_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:index]
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output", type=Path, default=Path("gemma4_baseline.jsonl"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    loader = MathDatasetLoader()
    dataset = loader.load_data(
        eval_size=args.eval_size,
        seed=args.seed,
    )
    evaluation_set = dataset["test"].select(
        range(min(args.num_samples, len(dataset["test"])))
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map=args.device_map,
    )
    model.eval()

    correct = 0
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")

    for index, example in enumerate(evaluation_set):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        ).to(model.device)
        input_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_tokens = output[0][input_length:]
        response = processor.decode(generated_tokens, skip_special_tokens=True)
        predicted_boxed = extract_boxed(response)
        expected = extract_boxed(example["solution"])
        gold_expression = parse(f"\\boxed{{{expected}}}") if expected else []
        predicted_expression = parse(response)
        is_correct = bool(
            gold_expression
            and predicted_expression
            and verify(gold_expression, predicted_expression)
        )
        correct += int(is_correct)

        record = {
            "index": index,
            "problem": example["problem"],
            "level": example["level"],
            "type": example["type"],
            "expected_answer": expected,
            "predicted_answer": (
                str(predicted_expression[0]) if predicted_expression else None
            ),
            "predicted_boxed_answer": predicted_boxed,
            "boxed_format_compliant": predicted_boxed is not None,
            "correct": is_correct,
            "response": response,
        }
        records.append(record)
        with args.output.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{index + 1}/{len(evaluation_set)}] "
            f"correct={is_correct} running_accuracy={correct / (index + 1):.2%}",
            flush=True,
        )

    group_counts = defaultdict(lambda: {"correct": 0, "samples": 0})
    for record in records:
        for group_name in ("level", "type"):
            group = group_counts[group_name, record[group_name]]
            group["samples"] += 1
            group["correct"] += int(record["correct"])

    breakdown = {"level": {}, "type": {}}
    for (group_name, group_value), counts in sorted(group_counts.items()):
        breakdown[group_name][group_value] = {
            **counts,
            "accuracy": counts["correct"] / counts["samples"],
        }

    summary = {
        "model_id": args.model_id,
        "dataset_id": loader.dataset_name,
        "evaluation_fingerprint": evaluation_set._fingerprint,
        "samples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "eval_size": args.eval_size,
        "seed": args.seed,
        "thinking": args.thinking,
        "max_new_tokens": args.max_new_tokens,
        "boxed_format_compliance": (
            sum(record["boxed_format_compliant"] for record in records) / len(records)
        ),
        "results_file": str(args.output),
        "breakdown": breakdown,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
