import json
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile, TemporaryDirectory
from traceback import format_exc

import numpy as np
import requests

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30
_MAX_CHAR_DISPLAY = 2048
CLI_ARG_SIZE_LIMIT = 1024 * 3


def _check_executor_alive(executor_url: str) -> bool:
    try:
        status = requests.get(executor_url + "/", timeout=2).status_code
        return status in (200, 404)
    except Exception:
        return False


def _remote_code_exec_ces(code: str, stdin: str | None = None) -> tuple[bool, str]:
    ces_url = os.environ.get("CODER1_CES_URL", "http://localhost:8000")
    timeout = int(os.environ.get("CODER1_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS))
    max_retry_on_timeout = int(os.environ.get("CODER1_TIMEOUT_RETRIES", 2))
    cur_retry_on_timeout = 0

    while True:
        try:
            t_start = time.time()
            headers = {"Content-Type": "application/json"}
            resp = requests.post(
                ces_url + "/py_exec",
                data=json.dumps({"code": code, "timeout": timeout, "stdin": stdin}),
                headers=headers,
                timeout=timeout + 10,
            )
            status_line, outs = resp.text.split("\n", 1)
            succ_exit = status_line == "0"

            timed_out = (not succ_exit) and outs == "" and (time.time() - t_start > timeout)
            if timed_out:
                if cur_retry_on_timeout >= max_retry_on_timeout:
                    return False, _ERROR_MSG_PREFIX + f"Timeout for {timeout}s after retries"
                cur_retry_on_timeout += 1
                time.sleep(random.randint(5, 20))
                continue

            return succ_exit, outs
        except Exception:
            if not _check_executor_alive(ces_url):
                time.sleep(3)
                continue
            return False, _ERROR_MSG_PREFIX + format_exc()


def _code_exec_firejail(
    code: str,
    stdin: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    pytest_code: str | None = None,
) -> tuple[bool, str]:
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.pop("PYTHONPATH", None)

    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",
        "--rlimit-as=4096m",
        f"--timeout=00:00:{timeout}",
    ]

    try:
        if pytest_code is not None:
            with TemporaryDirectory() as tmpdir:
                if stdin is not None:
                    return False, _ERROR_MSG_PREFIX + "STDIN is not supported with pytest mode"

                solution_path = os.path.join(tmpdir, "solution.py")
                test_path = os.path.join(tmpdir, "test_solution.py")
                with open(solution_path, "w", encoding="utf-8") as f:
                    f.write(code)
                with open(test_path, "w", encoding="utf-8") as f:
                    f.write(pytest_code)

                command.insert(4, f"--whitelist={tmpdir}")
                command.extend(["python3", "-m", "pytest", tmpdir])
                result = subprocess.run(
                    command,
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                )
        else:
            if len(code) < CLI_ARG_SIZE_LIMIT:
                command.extend(["python3", "-c", code])
                result = subprocess.run(
                    command,
                    input=stdin.encode() if stdin else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                )
            else:
                with NamedTemporaryFile() as tmp:
                    tmp.write(code.encode())
                    tmp.flush()
                    command.insert(4, f"--whitelist={tmp.name}")
                    command.extend(["python3", tmp.name])
                    result = subprocess.run(
                        command,
                        input=stdin.encode() if stdin else None,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                        check=False,
                    )

        stdout = result.stdout.decode()
        stderr = result.stderr.decode().strip()
        if result.returncode == 0:
            return True, stdout
        return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    except FileNotFoundError:
        return False, _ERROR_MSG_PREFIX + "firejail not found"
    except Exception:
        return False, _ERROR_MSG_PREFIX + format_exc()


def _code_exec(code: str, stdin: str | None = None, pytest_code: str | None = None) -> tuple[bool, str]:
    backend = os.environ.get("CODER1_EXEC", "firejail").lower()
    if backend == "ces":
        return _remote_code_exec_ces(code=code, stdin=stdin)
    if backend == "firejail":
        timeout = int(os.environ.get("CODER1_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS))
        return _code_exec_firejail(code=code, stdin=stdin, timeout=timeout, pytest_code=pytest_code)
    return False, _ERROR_MSG_PREFIX + f"Unknown CODER1_EXEC backend: {backend}"


def _validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>$", re.DOTALL)
    return bool(pattern.match(processed_str.strip()))


def _try_extract_solution(solution_str: str) -> str:
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return solution_str


_CODE_PATTERN = re.compile(r"```(?:\\w+)?\\n(.*?)\\n```", re.DOTALL)


def _extract_code_from_string(solution_str: str) -> str:
    answer_str = _try_extract_solution(solution_str)
    code_blocks = _CODE_PATTERN.findall(answer_str)
    return "\n".join(code_blocks).strip()


def _remote_check_stdio(code: str, stdin: str, stdout: str) -> tuple[bool, str, str, str]:
    succ, output = _code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout


def _compute_score(
    solution_str: str,
    ground_truth: str | dict,
    extra_info: dict,
    format_reward: float = 0.1,
    answer_reward: float = 1.0,
) -> tuple[float, str]:
    reward_log = []
    pass_fmt = _validate_response_structure(solution_str)
    solution_code = _extract_code_from_string(solution_str)

    if (not pass_fmt) or len(solution_code) == 0:
        reward_log.append("bad format or empty code block")
        return -answer_reward - format_reward, "\n".join(reward_log)

    if isinstance(ground_truth, str):
        parsed_ground_truth = json.loads(ground_truth)
    else:
        parsed_ground_truth = ground_truth

    t_start = time.time()
    output = ""

    if "functional" in parsed_ground_truth:
        succ, output = _code_exec(solution_code + "\n" + parsed_ground_truth["functional"])
        if not succ:
            reward_log.append(f"functional test failed in {time.time() - t_start:.1f}s")
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            return format_reward, "\n".join(reward_log)

    elif "pytest" in parsed_ground_truth:
        succ, output = _code_exec(solution_code, pytest_code=parsed_ground_truth["pytest"])
        if not succ:
            reward_log.append(f"pytest test failed in {time.time() - t_start:.1f}s")
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            return format_reward, "\n".join(reward_log)

    elif "inputs" in parsed_ground_truth and "outputs" in parsed_ground_truth:
        stdin_list = parsed_ground_truth["inputs"]
        stdout_list = parsed_ground_truth["outputs"]

        with ThreadPoolExecutor(max_workers=min(8, len(stdin_list))) as executor:
            futures = [
                executor.submit(_remote_check_stdio, solution_code, stdin, stdout)
                for stdin, stdout in zip(stdin_list, stdout_list, strict=True)
            ]
            for future in as_completed(futures):
                succ, current_output, stdin, stdout = future.result()
                if (not succ) or current_output.strip() != stdout.strip():
                    got = current_output[:_MAX_CHAR_DISPLAY]
                    reward_log.append(f"stdio test failed in {time.time() - t_start:.1f}s")
                    reward_log.append(f"input={stdin!r}")
                    reward_log.append(f"expect={stdout.strip()!r}")
                    reward_log.append(f"got={got.strip()!r}")
                    return format_reward, "\n".join(reward_log)
                output = current_output
    else:
        raise ValueError("unsupported ground_truth format")

    reward_log.append("test execution passed")
    if output:
        reward_log.append(output[:_MAX_CHAR_DISPLAY])
    return format_reward + answer_reward, "\n".join(reward_log)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    format_reward: float = 0.1,
    answer_reward: float = 1.0,
):
    if data_source != "code":
        # Keep non-code samples neutral in this coder1-only reward.
        return 0.0

    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    if extra_info is None:
        extra_info = {}

    score, reward_log = _compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        format_reward=format_reward,
        answer_reward=answer_reward,
    )

    verbose = os.environ.get("CODER1_REWARD_VERBOSE", "0") == "1"
    if verbose:
        print(f"[coder1 reward] score={score}\n{reward_log}\n")

    return score
