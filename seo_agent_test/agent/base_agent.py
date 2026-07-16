"""
Generic tool-using agent loop, built on Groq's OpenAI-compatible chat
completions API. Each specialist, the planner, the synthesizer, and the
critic are all instances of ToolAgent configured with a different system
prompt / tool set / model -- this is the one piece of "agent runtime" the
rest of the system builds on.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

import groq
from groq import Groq

from . import tools
from .config import DEFAULT_MODEL, MAX_TOOL_ITERATIONS, RATE_LIMIT_MAX_RETRIES

TOOL_IMPL: dict[str, Callable[[dict], dict]] = {
    "fetch_page": lambda args: tools.fetch_page(args["url"]),
    "parse_seo_elements": lambda args: tools.parse_seo_elements(args["html"], args["base_url"]),
    "fetch_robots_txt": lambda args: tools.fetch_robots_txt(args["domain_or_url"]),
    "fetch_sitemap": lambda args: tools.fetch_sitemap(args["domain_or_url"]),
    "check_ssl_certificate": lambda args: tools.check_ssl_certificate(args["domain_or_url"]),
    "analyze_security_headers": lambda args: tools.analyze_security_headers(args["headers"]),
    "check_links_status": lambda args: tools.check_links_status(args["urls"]),
}

_WAIT_TIME_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _seconds_to_wait(error: groq.RateLimitError, attempt: int) -> float:
    """Parse Groq's suggested wait time out of the error message if present,
    otherwise fall back to exponential backoff."""
    message = str(getattr(error, "message", "") or str(error))
    match = _WAIT_TIME_RE.search(message)
    if match:
        return float(match.group(1)) + 0.5  # small safety buffer
    return min(2 ** attempt, 30)


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


class ToolAgent:
    """A single named agent with its own system prompt, tool set, and loop."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        client_tools: list[dict] | None = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.client_tools = client_tools or []
        self.model = model
        self.max_iterations = max_iterations
        self.log = log_fn or (lambda msg: None)
        self._client = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = Groq()  # picks up GROQ_API_KEY from env
        return self._client

    def _call_with_retry(self, **kwargs):
        """Call the Groq API, automatically retrying on rate-limit (429)
        errors using the wait time Groq suggests in the error message."""
        last_error = None
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except groq.RateLimitError as e:
                last_error = e
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    break
                wait = _seconds_to_wait(e, attempt)
                self.log(f"[{self.name}] rate limited, retrying in {wait:.1f}s "
                          f"(attempt {attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
                time.sleep(wait)
        raise last_error

    def run(self, user_message: str, expect_json: bool = True) -> Any:
        """Run the agentic loop to completion and return parsed JSON (or raw text)."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _ in range(self.max_iterations):
            kwargs = dict(model=self.model, messages=messages, max_tokens=4096, temperature=0.2)
            if self.client_tools:
                kwargs["tools"] = self.client_tools
                kwargs["tool_choice"] = "auto"

            response = self._call_with_retry(**kwargs)
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            # Append the assistant turn (Groq/OpenAI format expects tool_calls
            # to be echoed back verbatim when present).
            assistant_turn: dict = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_turn)

            if not tool_calls:
                final_text = message.content or ""
                if not expect_json:
                    return final_text
                try:
                    return _extract_json(final_text)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"[{self.name}] model did not return valid JSON:\n{final_text}"
                    ) from e

            for tc in tool_calls:
                self.log(f"[{self.name}] -> {tc.function.name}({tc.function.arguments[:150]})")
                try:
                    args = json.loads(tc.function.arguments)
                    result = TOOL_IMPL[tc.function.name](args)
                except Exception as e:
                    result = {"ok": False, "error": f"Tool execution failed: {e}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)[:15000],
                    }
                )

        raise RuntimeError(f"[{self.name}] exceeded {self.max_iterations} tool-call iterations.")
