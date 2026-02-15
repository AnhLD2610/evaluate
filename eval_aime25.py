from transformers import AutoTokenizer
import json
import numpy as np
from datasets import load_dataset
from dataset import extract_answer_llm, extract_answer_qwq, is_float
from vllm import LLM, SamplingParams
import argparse


def check_answer_aime(response, gt_answer):
    """Check if a single response is correct for AIME (integer answers)."""
    llm_answer = extract_answer_qwq(response)
    if not is_float(llm_answer):
        llm_answer = extract_answer_llm(response)

    is_correct = False
    if is_float(gt_answer) and is_float(llm_answer):
        try:
            is_correct = (int(round(float(gt_answer))) == int(round(float(llm_answer))))
        except OverflowError:
            is_correct = False

    return is_correct, llm_answer


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k on AIME 2025")
    parser.add_argument("--model_name", type=str, default="Zigeng/R1-VeriThinker-7B")
    parser.add_argument("--tp", type=int, default=4, help="tensor_parallel_size")
    parser.add_argument("--n", type=int, default=128, help="Number of samples per prompt (max k)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=40000)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    test_examples = load_dataset("math-ai/aime25", split="test")
    test_examples = list(test_examples)

    seed = 42

    llm = LLM(model=args.model_name, tensor_parallel_size=args.tp, max_model_len=args.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    results_per_question = []

    batch_size = 30

    for i in range(0, len(test_examples), batch_size):
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=seed + i,
            stop=["\n</think>"],
            n=args.n,
        )

        end = min(i + batch_size, len(test_examples))
        batch_examples = test_examples[i:end]

        batch_prompts = []
        batch_gt_answers = []

        for example in batch_examples:
            prompt = example["problem"]
            tail = r" Please reason step by step, and put your final answer within \boxed{}."
            messages = [{"role": "user", "content": prompt + tail}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(text)
            batch_gt_answers.append(example["answer"])

        outputs = llm.generate(batch_prompts, sampling_params)

        for j, output in enumerate(outputs):
            gt_answer = batch_gt_answers[j]
            sample_correct = []

            for s_idx, single_output in enumerate(output.outputs):
                response = single_output.text
                is_correct, prediction = check_answer_aime(response, gt_answer)
                sample_correct.append(is_correct)

            results_per_question.append({
                "question": batch_examples[j]["problem"],
                "gt_answer": gt_answer,
                "n": args.n,
                "num_correct": sum(sample_correct),
                "sample_correct": sample_correct,
            })

            current_idx = i + j + 1
            print(f"[{current_idx}/{len(test_examples)}] correct: {sum(sample_correct)}/{args.n}")

    # --- Compute pass@k (naive): solved if ANY of first k samples is correct ---
    k_values = [k for k in [1, 4, 8, 16, 32, 64, 128] if k <= args.n]

    print("\n" + "=" * 60)
    print(f"AIME 2025 | Model: {args.model_name} | n={args.n}")
    print("=" * 60)

    pass_k_results = {}
    for k in k_values:
        solved = 0
        for r in results_per_question:
            if any(r["sample_correct"][:k]):
                solved += 1
        acc = solved / len(results_per_question)
        pass_k_results[f"pass@{k}"] = acc
        print(f"  pass@{k:>3d}: {acc:.4f}  ({acc*100:.2f}%)  [{solved}/{len(results_per_question)}]")

    print("=" * 60)

    output_path = args.output or f"aime25_pass_at_k_n{args.n}.json"
    save_data = {
        "model": args.model_name,
        "n": args.n,
        "num_questions": len(results_per_question),
        "pass_at_k": pass_k_results,
        "per_question": results_per_question,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()