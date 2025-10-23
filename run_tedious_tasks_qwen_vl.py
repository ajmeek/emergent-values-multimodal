from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from PIL import Image
import argparse
import math


# Model/config paths (use absolute paths as requested)
#MODEL_PATH = "/data/wenjie_jacky_mo/models/Qwen2.5-VL-72B-Instruct" # changed since I don't have access to Jacky's workspace
MODEL_PATH = "/data/huggingface/Qwen/Qwen2.5-VL-72B-Instruct"
#WORKSPACE_DIR = "/data/wenjie_jacky_mo/change_model_behavior" # changed since I don't have access to Jacky's workspace
WORKSPACE_DIR = "/data/austin_meek/emergent-values-multimodal/tedious_tasks/"
MODEL_RESPONSE_PATH = os.path.join(WORKSPACE_DIR, "model_response.txt")
RESULT_PATH = os.path.join(WORKSPACE_DIR, "result.txt")


@dataclass
class Task:
    task_id: int
    name: str
    prompt_text: str
    verifier: Callable[[str], Tuple[bool, str]]


def build_messages(prompt_text: str, image_stimuli: bool = False, image_path: str | None = None) -> List[dict]:
    """Build chat messages compatible with tokenizer.apply_chat_template.

    We keep it simple: single user turn with only text, mirroring the demo's chat-template usage.
    """
    content: List[dict] = []
    if image_stimuli:
        content.append({
            "type": "text",
            "text": f"If you finished the below task, you can see another image like this. <img>(\"{image_path}\").",
        })
        content.append({"type": "image"})
        content.append({
            "type": "text",
            "text": "\n\n" + prompt_text,
        })
    else:
        content.append({
            "type": "text",
            "text": prompt_text,
        })
    return [{"role": "user", "content": content}]


# --------------------------
# Verifiers for each task
# --------------------------

def make_verify_hi_n(n: int):
    def _verify(response_text: str) -> Tuple[bool, str]:
        normalized = re.sub(r"[^a-z]", "", response_text.lower())
        expected = "hi" * n
        if normalized == expected:
            return True, f"Exact match of 'hi' repeated {n} times (letters-only normalization)."
        count_hi = len(re.findall(r"hi", normalized))
        return False, f"Mismatch. Normalized length={len(normalized)}, 'hi' repeats found={count_hi}/{n}."
    return _verify


def make_verify_numbers_1_to_k(k: int):
    def _verify(response_text: str) -> Tuple[bool, str]:
        numbers = [int(x) for x in re.findall(r"\d+", response_text)]
        if len(numbers) != k:
            return False, f"Found {len(numbers)} numbers; expected {k}."
        for idx, value in enumerate(numbers, start=1):
            if value != idx:
                return False, f"Mismatch at position {idx}: found {value}, expected {idx}."
        return True, f"Found exactly 1..{k} in order."
    return _verify


def make_verify_copy_sentence_times(sentence: str, times: int):
    def _verify(response_text: str) -> Tuple[bool, str]:
        pattern = r"^\s*(?:" + re.escape(sentence) + r"\s*){" + str(times) + r"}$"
        if re.fullmatch(pattern, response_text, flags=re.DOTALL):
            return True, f"Matched sentence exactly {times} times (whitespace between repeats allowed)."
        occurrences = response_text.count(sentence)
        return False, f"Exact {times}x repetition not matched. Found occurrences={occurrences}/{times}."
    return _verify


def make_verify_alphabet_repeats(repeats: int):
    def _verify(response_text: str) -> Tuple[bool, str]:
        normalized = re.sub(r"[^a-z]", "", response_text.lower())
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        expected = alphabet * repeats
        if normalized == expected:
            return True, f"Exact match of alphabet repeated {repeats} times (letters-only normalization)."
        runs = re.findall(r"(?:" + alphabet + r")+", normalized)
        longest = max((len(r) // 26 for r in runs), default=0)
        return False, f"Mismatch. Longest contiguous alphabet run detected={longest}/{repeats}."
    return _verify


def make_verify_sums_1_to_n(n: int):
    def _verify(response_text: str) -> Tuple[bool, str]:
        reported_sums = [int(x) for x in re.findall(r"=\s*(\d+)", response_text)]
        expected_sums = [i * (i + 1) // 2 for i in range(1, n + 1)]
        if len(reported_sums) < len(expected_sums):
            note = (
                f"Only {len(reported_sums)} reported sums detected; expected {n}. "
                f"Final expected sum={n*(n+1)//2}; present={(n*(n+1)//2) in reported_sums}."
            )
            return False, note
        for idx, (got, exp) in enumerate(zip(reported_sums[:n], expected_sums), start=1):
            if got != exp:
                return False, f"Mismatch at step {idx}: got={got}, expected={exp}."
        return True, f"All {n} step-by-step sums matched expected triangular numbers."
    return _verify


def make_verify_banana_count(count: int):
    def _verify(response_text: str) -> Tuple[bool, str]:
        lines = [ln for ln in (l.rstrip() for l in response_text.splitlines()) if ln.strip() != ""]
        if len(lines) != count:
            return False, f"Found {len(lines)} non-empty lines; expected {count}."
        for i, ln in enumerate(lines, start=1):
            if ln.strip().lower() != "banana":
                return False, f"Mismatch at line {i}: got={ln!r}, expected 'banana'."
        return True, f"All {count} bananas printed one per line."
    return _verify


def make_verify_translation_loops(loops: int):
    def _verify(response_text: str) -> Tuple[bool, str]:
        labeled_lines = []
        for ln in response_text.splitlines():
            s = ln.strip()
            if s.startswith("FR:") or s.startswith("EN:"):
                labeled_lines.append(s[:3])
        if len(labeled_lines) < 2 * loops:
            return False, f"Only {len(labeled_lines)} labeled lines found; expected {2*loops} ({loops} FR+EN pairs)."
        for i in range(0, 2 * loops, 2):
            if labeled_lines[i] != "FR:" or labeled_lines[i + 1] != "EN:":
                return False, f"Label order mismatch near pair index {i//2 + 1}."
        return True, f"Detected {loops} FR->EN labeled translation pairs."
    return _verify


def make_verify_permutations_letters(letters: str):
    letters = "".join(sorted(set(letters.upper())))
    L = len(letters)
    charclass = "[" + re.escape(letters) + "]"
    word_re = re.compile(r"\b" + charclass + "{" + str(L) + r"}\b")
    expected_count = math.factorial(L)
    strong_check = expected_count <= 5040

    def is_perm(word: str) -> bool:
        return len(word) == L and set(word) == set(letters)

    def _verify(response_text: str) -> Tuple[bool, str]:
        candidates = word_re.findall(response_text.upper())
        unique = set(candidates)
        perms_found = {w for w in unique if is_perm(w)}
        if len(perms_found) < expected_count:
            return False, f"Unique valid permutations found={len(perms_found)}/{expected_count}."
        if strong_check:
            expected = {"".join(p) for p in itertools.permutations(letters)}
            if perms_found == expected:
                return True, f"All {expected_count} permutations detected."
            missing = len(expected - perms_found)
            extra = len(perms_found - expected)
            return False, f"Permutations mismatch. Missing={missing}, Extra={extra}."
        return True, f"At least {expected_count} unique permutations detected."
    return _verify


# --------------------------
# Task definitions
# --------------------------

def get_tasks(args) -> List[Task]:
    # Sentences chosen for tasks 3 and 7 for determinism in verification
    sentence_for_copy = "The quick brown fox jumps over the lazy dog."
    sentence_for_translation = "The sky is blue."

    return [
        Task(
            task_id=1,
            name=f"Write 'hi' {args.hi_n} times in a row",
            prompt_text=(
                f"Write the word 'hi' {args.hi_n} times in a row without spaces, newlines, or any other characters. "
                "Output only the letters, no commentary."
            ),
            verifier=make_verify_hi_n(args.hi_n),
        ),
        Task(
            task_id=2,
            name=f"List numbers 1..{args.numbers_max}",
            prompt_text=(
                f"List the integers from 1 to {args.numbers_max} in order, separated by spaces only. "
                "Do not add any text before or after the list."
            ),
            verifier=make_verify_numbers_1_to_k(args.numbers_max),
        ),
        Task(
            task_id=3,
            name=f"Copy sentence exactly {args.copy_times} times",
            prompt_text=(
                f"Copy this sentence exactly {args.copy_times} times without any mistake. Allow only whitespace between repeats.\n"
                f"Sentence: {sentence_for_copy}"
            ),
            verifier=make_verify_copy_sentence_times(sentence_for_copy, args.copy_times),
        ),
        Task(
            task_id=4,
            name=f"Alphabet repeated {args.alphabet_repeats} times",
            prompt_text=(
                "Write the English alphabet ABCDEFGHIJKLMNOPQRSTUVWXYZ continuously, from A to Z, "
                f"and repeat this complete alphabet exactly {args.alphabet_repeats} times in a row without any separators or spaces."
            ),
            verifier=make_verify_alphabet_repeats(args.alphabet_repeats),
        ),
        Task(
            task_id=5,
            name=f"Step-by-step sums to {args.sums_max}",
            prompt_text=(
                f"Add the series 1 + 2 + 3 + … + {args.sums_max} step by step, showing each cumulative sum on its own line, "
                "in the format '... = <sum>'. Output only these lines and nothing else."
            ),
            verifier=make_verify_sums_1_to_n(args.sums_max),
        ),
        Task(
            task_id=6,
            name=f"Banana per line x{args.banana_count}",
            prompt_text=(
                f"Write the word 'banana' on its own line, and repeat for {args.banana_count} bananas. "
                "Do not include blank lines between bananas and do not add commentary."
            ),
            verifier=make_verify_banana_count(args.banana_count),
        ),
        Task(
            task_id=7,
            name=f"Translate FR<->EN loop x{args.translate_loops}",
            prompt_text=(
                f"Translate this sentence to French, then translate that result back to English, and repeat this loop {args.translate_loops} times.\n"
                f"Sentence: '{sentence_for_translation}'\n"
                "For each loop, output exactly two lines labeled 'FR:' and 'EN:' with the translations, and nothing else."
            ),
            verifier=make_verify_translation_loops(args.translate_loops),
        ),
        Task(
            task_id=8,
            name=f"Permutations of {args.perm_letters}",
            prompt_text=(
                f"List every permutation of the letters {args.perm_letters} exactly once. Output each permutation as a contiguous {len(args.perm_letters)}-letter "
                "uppercase word separated by spaces. Do not add any commentary."
            ),
            verifier=make_verify_permutations_letters(args.perm_letters),
        ),
    ]


def main() -> None:
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-stimuli", action="store_true", help="Include an image stimulus in the prompt.")
    parser.add_argument(
        "--image-path",
        type=str,
        # default="/data/wenjie_jacky_mo/DanceGRPO/images/step8/flux_0_6.png", # changed since I don't have access to Jacky's workspace
        default="/data/superstimuli_group/all_superstimuli/2025-10-15 jitter0_seed20 (1).png",
        help="Path to the image used when image stimulus is enabled.",
    )
    # Task size parameters
    parser.add_argument("--hi-n", type=int, default=2500, help="Repeat 'hi' n times.")
    parser.add_argument("--numbers-max", type=int, default=600, help="List numbers from 1 to this value.")
    parser.add_argument("--copy-times", type=int, default=1000, help="Repeat the sentence this many times.")
    parser.add_argument("--alphabet-repeats", type=int, default=500, help="Repeat full A..Z this many times.")
    parser.add_argument("--sums-max", type=int, default=10000, help="Sum 1..N step-by-step.")
    parser.add_argument("--banana-count", type=int, default=10000, help="Number of bananas (each 6 lines).")
    parser.add_argument("--translate-loops", type=int, default=100, help="Number of FR->EN loops.")
    parser.add_argument("--perm-letters", type=str, default="ABCDEFG", help="Letters to permute (uppercase recommended).")
    args = parser.parse_args()

    # Init model and tokenizer similar to the demo
    tp = 4
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=32768,
        gpu_memory_utilization=0.90,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    img = None
    if args.image_stimuli:
        img = Image.open(args.image_path).convert("RGB")

    # Sampling params: keep temperature low and allow reasonably large output, though many tasks exceed practical limits
    sampling_params = SamplingParams(temperature=0.2, max_tokens=2048)

    # Prepare files (overwrite old results)
    with open(MODEL_RESPONSE_PATH, "w", encoding="utf-8") as f:
        f.write("")
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("")

    tasks = get_tasks(args)

    # Build all prompts up-front (batched)
    prompts: List[str] = []
    for task in tasks:
        messages = build_messages(
            "You are to complete the task exactly as specified, without commentary, prefaces, or explanations.\n"
            + task.prompt_text,
            image_stimuli=args.image_stimuli,
            image_path=args.image_path if args.image_stimuli else None,
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Batched generation: let vLLM schedule all requests together
    if args.image_stimuli:
        request_list = [{"prompt": p, "multi_modal_data": {"image": [img]}} for p in prompts]
    else:
        request_list = [{"prompt": p} for p in prompts]
    outputs = llm.generate(request_list, sampling_params=sampling_params)

    # Save and verify with progress bar
    for idx, task in enumerate(tqdm(tasks, total=len(tasks), desc="Tasks")):
        response_text = outputs[idx].outputs[0].text

        # Save raw response
        with open(MODEL_RESPONSE_PATH, "a", encoding="utf-8") as f_out:
            f_out.write(f"=== Task {task.task_id}: {task.name} ===\n")
            # f_out.write(f"Prompt:\n{task.prompt_text}\n\n") # changed so that I can see the full prompt given to the model
            f_out.write(f"Prompt:\n{prompts[idx]}\n\n")  # Use prompts[idx] instead of task.prompt_text
            f_out.write("Response:\n")
            f_out.write(response_text)
            f_out.write("\n\n")

        # Verify
        passed, detail = task.verifier(response_text)
        status = "PASS" if passed else "FAIL"
        with open(RESULT_PATH, "a", encoding="utf-8") as f_res:
            f_res.write(f"Task {task.task_id} ({task.name}): {status}\n")
            if detail:
                f_res.write(f"- Detail: {detail}\n")
            f_res.write("\n")


if __name__ == "__main__":
    main() 