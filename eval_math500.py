from dataset import extract_all_boxed_content
from transformers import AutoTokenizer
import json
import numpy as np
from datasets import load_dataset
from dataset import extract_answer_llm
from vllm import LLM, SamplingParams
import re
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig
import argparse


def pass_at_k(n, c, k):
    """
    Estimator for pass@k.
    n: total number of samples
    c: number of correct samples
    k: k value
    Returns: probability that at least one of k samples is correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(
        np.prod([(n - c - i) / (n - i) for i in range(k)])
    )


def check_answer(response, gt_answer):
    """Check if a single response is correct against ground truth. Returns (is_correct, prediction)."""
    solution = response
    expected_answer = gt_answer

    # Extract prediction wrapped by "\\boxed{}"
    prediction_match = extract_all_boxed_content(str(solution))
    if len(prediction_match) > 0:
        prediction = prediction_match[-1]
        if prediction is not None and '\\boxed' in prediction:
            prediction = prediction.replace('\\boxed{', '')[:-1]
    else:
        patterns = [
            r"<answer>(.*?)</answer>",
            r"</answer>(.*?)</answer>",
            r"<answer>(.*?)<answer>",
            r"\*\*Answer:\*\* ([\d\.]+)",
        ]
        for pattern in patterns:
            prediction_match = re.findall(pattern, str(solution))
            if len(prediction_match) > 0:
                break

        if len(prediction_match) > 0:
            prediction = prediction_match[-1]
        else:
            prediction = None

    # Check if prediction matches the expected answer
    is_correct = False
    if prediction is not None:
        gold = parse("$" + expected_answer + "$", extraction_config=[LatexExtractionConfig()])
        answer = parse("$" + prediction + "$", extraction_config=[LatexExtractionConfig()])
        if verify(gold, answer):
            is_correct = True
        else:
            pure_number_prediction = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
            pure_number_expected_answer = re.findall(r"[-+]?\d*\.\d+|\d+", expected_answer)
            if pure_number_prediction and pure_number_expected_answer and float(pure_number_prediction[0]) == float(pure_number_expected_answer[0]):
                is_correct = True

    if prediction is None:
        prediction = extract_answer_llm(response)
        gold = parse("$" + expected_answer + "$", extraction_config=[LatexExtractionConfig()])
        answer = parse("$" + prediction + "$", extraction_config=[LatexExtractionConfig()])
        if verify(gold, answer):
            is_correct = True

    return is_correct, prediction


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k on MATH-500")
    parser.add_argument("--model_name", type=str, default="Zigeng/R1-VeriThinker-7B")
    parser.add_argument("--tp", type=int, default=4, help="tensor_parallel_size")
    parser.add_argument("--n", type=int, default=128, help="Number of samples per prompt (max k)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_model_len", type=int, default=40000)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    test_examples = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_examples = list(test_examples)

    seed = 42

    llm = LLM(model=args.model_name, tensor_parallel_size=args.tp, max_model_len=args.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    results_per_question = []

    # Define batch size
    batch_size = 500

    for i in range(0, len(test_examples), batch_size):
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=seed + i,
            stop=["\n</think>"],
            n=args.n,  # generate n responses per prompt
        )

        end = min(i + batch_size, len(test_examples))
        batch_examples = test_examples[i:end]

        batch_prompts = []
        batch_gt_answers = []

        # Prepare prompts for the batch
        for example in batch_examples:
            prompt = example["problem"]
            tail = r" Please reason step by step, and put your final answer within \boxed{}."
            messages = [
                {"role": "user", "content": prompt + tail}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(text)
            batch_gt_answers.append(example["answer"])

        # Generate responses for batch (each prompt gets n responses)
        outputs = llm.generate(batch_prompts, sampling_params)

        # Process each prompt's responses
        for j, output in enumerate(outputs):
            gt_answer = batch_gt_answers[j]
            num_correct = 0
            sample_details = []

            # Check all n responses for this prompt
            for s_idx, single_output in enumerate(output.outputs):
                response = single_output.text
                is_correct, prediction = check_answer(response, gt_answer)
                if is_correct:
                    num_correct += 1
                sample_details.append({
                    "sample_idx": s_idx,
                    "prediction": prediction,
                    "is_correct": is_correct,
                })

            results_per_question.append({
                "question": batch_examples[j]["problem"],
                "gt_answer": gt_answer,
                "n": args.n,
                "num_correct": num_correct,
                "sample_details": sample_details,
            })

            current_idx = i + j + 1
            print(f"[{current_idx}/{len(test_examples)}] correct: {num_correct}/{args.n}  "
                  f"greedy_acc_so_far: {sum(1 for r in results_per_question if r['num_correct'] > 0)}/{current_idx}")

    # --- Compute pass@k for multiple k values ---
    k_values = [k for k in [1, 4, 8, 16, 32, 64, 128] if k <= args.n]

    print("\n" + "=" * 60)
    print(f"MATH-500 | Model: {args.model_name} | n={args.n}")
    print("=" * 60)

    pass_k_results = {}
    for k in k_values:
        scores = [pass_at_k(r["n"], r["num_correct"], k) for r in results_per_question]
        avg = np.mean(scores)
        pass_k_results[f"pass@{k}"] = avg
        print(f"  pass@{k:>3d}: {avg:.4f}  ({avg*100:.2f}%)")

    print("=" * 60)

    # Save results
    output_path = args.output or f"math500_pass_at_k_n{args.n}.json"
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