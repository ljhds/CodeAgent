import os
import re
from typing import Any
from uuid import uuid4

import requests

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer


@register("coder1_multiturn_agent")
class Coder1MultiTurnAgentLoop(AgentLoopBase):
    """Coder1 multi-turn loop with think/code/observation/answer protocol."""

    CODE_PATTERN = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        *args,
        max_turns: int = 4,
        max_obs_length: int = 1000,
        sandbox_url: str | None = None,
        invalid_retry_message: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        self.max_turns = max_turns
        self.max_obs_length = max_obs_length
        self.sandbox_url = sandbox_url or os.getenv("REWARD_CODE_SANDBOX_URL", "http://127.0.0.1:8090/run_code")
        self.invalid_retry_message = invalid_retry_message or (
            "<observation>Invalid action. Use <code>...</code> to run code or "
            "<answer>...</answer> to finish.</observation>"
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = (kwargs.get("multi_modal_data") or {}).get("image", None)

        metrics: dict[str, float] = {"generate_sequences": 0.0, "tool_calls": 0.0}
        request_id = uuid4().hex

        prompt_ids: list[int] = []
        response_ids: list[int] = []
        response_mask: list[int] = []

        want_logprobs = bool(sampling_params.get("logprobs", False))
        response_logprobs: list[float] | None = [] if want_logprobs else None

        for _ in range(self.max_turns):
            current_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            if len(current_prompt_ids) > self.prompt_length:
                current_prompt_ids = current_prompt_ids[-self.prompt_length :]
            if not prompt_ids:
                prompt_ids = list(current_prompt_ids)

            with simple_timer("generate_sequences", metrics):
                token_output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=current_prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                )

            raw_response = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(token_output.token_ids, skip_special_tokens=True),
            )
            processed_response, action = self._postprocess_response(raw_response)
            llm_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer(processed_response, add_special_tokens=False)["input_ids"],
            )
            self._append_tokens(
                response_ids,
                response_mask,
                response_logprobs,
                llm_ids,
                token_mask=1,
                llm_logprobs=token_output.log_probs,
            )
            messages.append({"role": "assistant", "content": processed_response})

            if action == "answer" or len(response_ids) >= self.response_length:
                break

            if action == "code":
                with simple_timer("tool_calls", metrics):
                    observation = await self.loop.run_in_executor(None, lambda: self._build_observation(processed_response))
            else:
                observation = self.invalid_retry_message

            obs_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer(observation, add_special_tokens=False)["input_ids"],
            )
            self._append_tokens(response_ids, response_mask, response_logprobs, obs_ids, token_mask=0)
            messages.append({"role": "user", "content": observation})

            if len(response_ids) >= self.response_length:
                break

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs is not None else None,
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=len(messages),
            metrics=metrics,
            extra_fields={},
        )
        return output

    def _append_tokens(
        self,
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        token_ids: list[int],
        token_mask: int,
        llm_logprobs: list[float] | None = None,
    ) -> None:
        remaining = self.response_length - len(response_ids)
        if remaining <= 0:
            return

        token_ids = token_ids[:remaining]
        response_ids.extend(token_ids)
        response_mask.extend([token_mask] * len(token_ids))

        if response_logprobs is None:
            return

        if token_mask == 1 and llm_logprobs and len(llm_logprobs) >= len(token_ids):
            response_logprobs.extend(llm_logprobs[: len(token_ids)])
        else:
            response_logprobs.extend([0.0] * len(token_ids))

    def _postprocess_response(self, text: str) -> tuple[str, str]:
        code_end = text.find("</code>")
        answer_end = text.find("</answer>")

        if code_end != -1:
            return text[: code_end + len("</code>")], "code"
        if answer_end != -1:
            return text[: answer_end + len("</answer>")], "answer"
        return text, "invalid"

    def _extract_last_code(self, text: str) -> str:
        matches = self.CODE_PATTERN.findall(text)
        if not matches:
            return ""
        return matches[-1].strip()

    def _build_observation(self, response_text: str) -> str:
        code = self._extract_last_code(response_text)
        if not code:
            return "<observation>No executable code found.</observation>"

        try:
            resp = requests.post(
                self.sandbox_url,
                json={"code": code, "language": "python"},
                timeout=30,
            )
            data = resp.json()
            run_result = data.get("run_result", {})
            stdout = str(run_result.get("stdout", ""))[: self.max_obs_length]
            stderr = str(run_result.get("stderr", ""))[: self.max_obs_length]
            return f"<observation>Code output: {stdout}\\nErrors: {stderr}</observation>"
        except Exception as exc:
            return f"<observation>Code execution failed: {exc}</observation>"
