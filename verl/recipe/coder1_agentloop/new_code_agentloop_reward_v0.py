import os
import re
from typing import Any

import requests


def _extract_answer(text: str) -> str:
    parts = text.split("<answer>")
    if len(parts) < 2:
        return ""
    answer = parts[-1].split("</answer>")[0]
    return answer.strip()


def _extract_code_blocks(text: str) -> list[str]:
    return re.findall(r"<code>(.*?)</code>", text, re.DOTALL | re.IGNORECASE)


def _is_valid_sequence(content: str) -> tuple[bool, str]:
    tags_to_check = ["think", "code", "observation", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"mismatch {tag} tags"

    split_pattern = r"(</?(?:think|code|observation|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"
    for part in parts:
        if not part.strip():
            continue

        if re.match(r"</?(?:think|code|observation|answer)>", part):
            if part == "<think>" and state in ["start", "observation"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<code>" and state == "after_think":
                state = "in_code"
            elif part == "</code>" and state == "in_code":
                state = "after_code"
            elif part == "<observation>" and state == "after_code":
                state = "in_observation"
            elif part == "</observation>" and state == "in_observation":
                state = "observation"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"unexpected tag {part} in state {state}"
        else:
            if state in ["in_think", "in_code", "in_observation", "in_answer"]:
                continue
            if state in ["start", "after_think", "after_code", "observation"] and part.strip():
                return False, f"unexpected content in state {state}"

    if state != "end":
        return False, f"incomplete sequence, ended in {state}"

    return True, "ok"


def _exec_code(code: str) -> tuple[str, str]:
    url = os.getenv("REWARD_CODE_SANDBOX_URL", "http://127.0.0.1:8090/run_code")
    try:
        response = requests.post(
            url,
            json={"code": code, "language": "python"},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        stdout = str(payload.get("run_result", {}).get("stdout", ""))
        stderr = str(payload.get("run_result", {}).get("stderr", ""))
        return stdout[:1000], stderr[:1000]
    except Exception as exc:
        return "", f"sandbox error: {exc}"


def _answer_reward(user_message: str, answer: str) -> float:
    base_url = os.getenv("CODER1_REWARD_LLM_BASE_URL")
    api_key = os.getenv("CODER1_REWARD_LLM_API_KEY")
    model = os.getenv("CODER1_REWARD_LLM_MODEL", "qwen3-32b")
    if not base_url or not api_key:
        return 0.0

    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        prompt = (
            "请根据执行过程判断是否成功解决问题。"
            "只输出'是'或'否'。\\n\\n"
            f"问题:\\n{user_message}\\n\\n执行过程:\\n{answer}"
        )
        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a strict evaluator."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        if content == "是":
            return 1.0
        if content == "否":
            return -1.0
        return 0.0
    except Exception:
        return 0.0


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
):
    _ = data_source
    _ = ground_truth

    is_valid, _ = _is_valid_sequence(solution_str)
    if is_valid:
        score = 0.5
        code_blocks = _extract_code_blocks(solution_str)
        if code_blocks:
            stdout, stderr = _exec_code(code_blocks[-1])
            if "error" in stderr.lower() or "traceback" in stderr.lower():
                score -= 0.5
            else:
                score += 0.5
                user_message = ""
                if extra_info and isinstance(extra_info, dict):
                    user_message = str(extra_info.get("user_message", ""))
                score += _answer_reward(user_message, solution_str)
        return float(score)

    format_score = 0.0
    compact = solution_str.replace("\n", "")

    if solution_str.startswith("<think>"):
        format_score += 0.1
    if solution_str.endswith("</answer>"):
        format_score += 0.1
    if "</think><answer>" in compact:
        format_score += 0.1
    if "<think>" in solution_str and "</think>" in solution_str:
        format_score += 0.02
    if "<code>" in solution_str and "</code>" in solution_str:
        format_score += 0.02
    if "<answer>" in solution_str and "</answer>" in solution_str:
        format_score += 0.02

    return float(format_score)
