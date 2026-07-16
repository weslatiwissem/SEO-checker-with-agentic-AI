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
from .config import DEFAULT_MODEL, MAX_TOOL_ITERATIONS, RATE_LIMIT_MAX_RETRIES, FALLBACK_MODEL

TOOL_IMPL: dict[str, Callable[[dict], dict]] = {
    "fetch_page": lambda args: tools.fetch_page(args["url"]),
    "parse_seo_elements": lambda args: tools.parse_seo_elements(args["url"]),
    "fetch_robots_txt": lambda args: tools.fetch_robots_txt(args["domain_or_url"]),
    "fetch_sitemap": lambda args: tools.fetch_sitemap(args["domain_or_url"]),
    "check_ssl_certificate": lambda args: tools.check_ssl_certificate(args["domain_or_url"]),
    "analyze_security_headers": lambda args: tools.analyze_security_headers(args["headers"]),
    "check_links_status": lambda args: tools.check_links_status(args["urls"]),
}

# Groq formats wait times as e.g. "11.065s", "6m53.856s", or "1h4m12.576s" --
# match all three shapes, not just plain seconds.
_WAIT_TIME_RE = re.compile(
    r"try again in (?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?P<s>[\d.]+)s", re.IGNORECASE
)

# If Groq's suggested wait exceeds this, don't sleep silently for a
# potentially very long time (daily-quota waits can be over an hour) --
# fail fast with a clear message instead.
MAX_AUTO_WAIT_SECONDS = 900  # 15 minutes

# Matches "Limit 6000 ... Requested 6783" style messages, used to figure out
# how aggressively to shrink an oversized request.
_SIZE_LIMIT_RE = re.compile(r"Limit (\d+).*?Requested (\d+)", re.IGNORECASE | re.DOTALL)
MAX_SHRINK_ATTEMPTS = 3


def _shrink_ratio_from_error(error) -> float:
    """Figure out roughly how much smaller the payload needs to be, based on
    the limit/requested numbers Groq reports. Falls back to a safe default
    if those numbers aren't present in the message."""
    message = str(getattr(error, "message", "") or str(error))
    match = _SIZE_LIMIT_RE.search(message)
    if match:
        limit, requested = int(match.group(1)), int(match.group(2))
        if requested > 0:
            return max(0.3, min(0.85, (limit / requested) * 0.8))  # extra safety margin
    return 0.6


def _shrink_largest_message(messages: list[dict]) -> bool:
    """Find the largest message content in the conversation and truncate it.
    Mutates messages in place. Returns False if there's nothing left worth
    shrinking (so the caller knows to give up)."""
    candidates = [
        i for i, m in enumerate(messages)
        if isinstance(m.get("content"), str) and len(m["content"]) > 500
    ]
    if not candidates:
        return False
    idx = max(candidates, key=lambda i: len(messages[i]["content"]))
    content = messages[idx]["content"]
    new_len = max(500, int(len(content) * 0.6))
    messages[idx]["content"] = content[:new_len] + "\n...[truncated to fit model request-size limits]"
    return True


def _seconds_to_wait(error: groq.RateLimitError, attempt: int) -> float | None:
    """Parse Groq's suggested wait time out of the error message if present,
    otherwise fall back to exponential backoff. Returns None if the wait
    exceeds MAX_AUTO_WAIT_SECONDS (caller should stop retrying)."""
    message = str(getattr(error, "message", "") or str(error))
    match = _WAIT_TIME_RE.search(message)
    if match:
        hours = float(match.group("h") or 0)
        minutes = float(match.group("m") or 0)
        seconds = float(match.group("s") or 0)
        total = hours * 3600 + minutes * 60 + seconds + 1.0  # small safety buffer
        return total if total <= MAX_AUTO_WAIT_SECONDS else None
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
        max_output_tokens: int = 2048,
        log_fn: Callable[[str], None] | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.client_tools = client_tools or []
        self.model = model
        self.max_iterations = max_iterations
        self.max_output_tokens = max_output_tokens
        self.log = log_fn or (lambda msg: None)
        self._client = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = Groq()  # picks up GROQ_API_KEY from env
        return self._client

    def _call_with_retry(self, **kwargs):
        """Call the Groq API, handling two distinct failure modes:
        - 429 rate/quota limit: prefer immediately switching to FALLBACK_MODEL
          (a separate quota pool, zero wait) over sleeping through the
          primary model's limit.
        - 413 (or 429-labeled "reduce your message size") request-too-large:
          no amount of waiting or model-switching helps here -- shrink the
          largest message in the conversation and retry.
        Any other status error is re-raised immediately, unmodified."""
        last_error = None
        fell_back = False
        shrink_attempts = 0

        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except groq.APIStatusError as e:
                last_error = e
                status = getattr(e, "status_code", None)
                message_text = str(getattr(e, "message", "") or str(e))

                too_large = status == 413 or "reduce your message size" in message_text.lower()
                if too_large:
                    if shrink_attempts >= MAX_SHRINK_ATTEMPTS or not _shrink_largest_message(kwargs["messages"]):
                        raise RuntimeError(
                            f"[{self.name}] request is still too large for model "
                            f"'{kwargs.get('model')}' after {shrink_attempts} shrink attempt(s). "
                            f"Groq's message: {message_text}"
                        ) from e
                    shrink_attempts += 1
                    self.log(f"[{self.name}] request too large for the model, shrinking payload "
                              f"and retrying (attempt {shrink_attempts}/{MAX_SHRINK_ATTEMPTS})...")
                    continue

                if status != 429:
                    raise  # not a rate/quota/size issue -- don't swallow unrelated errors

                current_model = kwargs.get("model")
                if FALLBACK_MODEL and not fell_back and current_model != FALLBACK_MODEL:
                    self.log(
                        f"[{self.name}] '{current_model}' rate/quota limited -- switching to "
                        f"fallback model '{FALLBACK_MODEL}' instead of waiting..."
                    )
                    kwargs["model"] = FALLBACK_MODEL
                    self.model = FALLBACK_MODEL  # sticky for the rest of this agent's run
                    fell_back = True
                    continue  # retry immediately, no sleep needed

                wait = _seconds_to_wait(e, attempt)

                if wait is None:
                    fallback_note = f" (even after falling back to '{FALLBACK_MODEL}')" if fell_back else ""
                    raise RuntimeError(
                        f"[{self.name}] hit a Groq rate/quota limit requiring a wait longer than "
                        f"{MAX_AUTO_WAIT_SECONDS // 60} minutes{fallback_note}. This usually means "
                        f"your daily token quota (TPD) is exhausted. Wait for it to reset or "
                        f"upgrade your tier at https://console.groq.com/settings/billing.\n\n"
                        f"Groq's message: {message_text}"
                    ) from e

                if attempt == RATE_LIMIT_MAX_RETRIES:
                    break
                self.log(f"[{self.name}] rate limited, waiting {wait:.1f}s "
                          f"(attempt {attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
                time.sleep(wait)
        raise last_error

    def run(self, user_message: str, expect_json: bool = True) -> Any:
        """Run the agentic loop to completion and return parsed JSON (or raw text)."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        json_repair_attempted = False

        for _ in range(self.max_iterations):
            kwargs = dict(model=self.model, messages=messages, max_tokens=self.max_output_tokens, temperature=0.2)
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
                except json.JSONDecodeError:
                    if not json_repair_attempted:
                        self.log(f"[{self.name}] response was not valid JSON (likely truncated) "
                                  f"-- asking it to resend a shorter, complete version...")
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your last response was not valid, complete JSON (it may have been "
                                "cut off). Resend ONLY a single valid, complete JSON object -- no "
                                "markdown fences, no extra text -- and be more concise (shorter "
                                "lists, shorter strings) so it fits without being truncated."
                            ),
                        })
                        json_repair_attempted = True
                        continue
                    raise RuntimeError(
                        f"[{self.name}] model did not return valid JSON after a repair attempt:\n{final_text}"
                    )

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