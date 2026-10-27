#!/usr/bin/env python3
"""llmspeed - a speed test for OpenAI-compatible LLM endpoints.

It measures what a user or an agent harness waits for: time to first token,
first visible content, tool-call readiness, prefill rate, output rate, and
end-to-end time.
"""

from __future__ import annotations

import argparse
import getpass
import http.client
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

VERSION = "0.1.0"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_RUNS = 3
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 256
LABEL_WIDTH = 20
CUSTOM_KEY_ENV = "OPENAI_API_KEY"

PROVIDERS = {
    "coralbricks": {
        "label": "CoralBricks",
        "base_url": "https://inference.coralbricks.ai/v1",
        "api_key_env": "CORAL_API_KEY",
        "models": ["deepseek-v4.1-flash-fast", "deepseek-v4.1-flash"],
    },
}
CUSTOM_PROVIDER = {
    "label": "Custom OpenAI-compatible endpoint",
    "base_url": "",
    "api_key_env": CUSTOM_KEY_ENV,
    "models": [],
}

NUMERIC_KEYS = (
    "ttft_ms",
    "first_content_ms",
    "first_usable_ms",
    "tool_ready_ms",
    "e2e_ms",
    "decode_ms",
    "inter_chunk_p50_ms",
    "inter_chunk_p95_ms",
    "inter_chunk_max_ms",
    "visible_stall_p95_ms",
    "prefill_tps",
    "output_tps",
    "reasoning_tps",
    "content_tps",
    "tpot_ms",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "cost_usd",
)
TOKEN_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
}

BUILTIN_SCENARIOS = [
    {
        "name": "chat",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": "Explain in two short paragraphs why tail latency "
                "matters for interactive applications.",
            }
        ],
    },
    {
        "name": "reasoning",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "A train leaves at 09:15 and travels 312 km at "
                "96 km/h. It stops for 12 minutes. At what time does it "
                "arrive? Show your work.",
            }
        ],
    },
    {
        "name": "tool-call",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "What is the weather in Kuala Lumpur right now?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    },
]


class SpeedError(Exception):
    """A request or configuration failed."""


def chat_url(base_url):
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def get_provider(name):
    if name == "custom":
        return CUSTOM_PROVIDER
    if name in PROVIDERS:
        return PROVIDERS[name]
    known = ", ".join([*PROVIDERS, "custom"])
    raise SpeedError(f'unknown provider "{name}". Known providers: {known}')


def list_models(base_url, api_key, timeout=30):
    """Return the model ids from GET /models."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json", "User-Agent": f"llmspeed/{VERSION}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=timeout
        ) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode("utf-8", "replace").strip()
        raise SpeedError(
            f"cannot read the model list (HTTP {error.code}): {detail[:300]}"
        ) from None
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        raise SpeedError(f"cannot read the model list from {url}: {error}") from None

    rows = payload.get("data") if isinstance(payload, dict) else payload
    models = [
        row["id"] for row in rows or [] if isinstance(row, dict) and row.get("id")
    ]
    if not models:
        raise SpeedError("the model list is empty")
    return models


def build_body(model, scenario, reasoning_effort=None):
    body = {key: value for key, value in scenario.items() if key != "name"}
    body["model"] = model
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    body.setdefault("messages", [])
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    if "max_completion_tokens" in body:
        body.pop("max_tokens", None)
    else:
        body.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    return body


def open_stream(url, body, api_key, timeout):
    """Send the request and return the open streaming response."""
    for attempt in (0, 1):
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": f"llmspeed/{VERSION}",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            with error:
                detail = error.read().decode("utf-8", "replace").strip()
            if (
                attempt == 0
                and error.code in (400, 422)
                and "stream_options" in detail
                and "stream_options" in body
            ):
                body = {k: v for k, v in body.items() if k != "stream_options"}
                continue
            hint = ""
            if error.code == 404:
                hint = " Check that base_url includes the API version, for example /v1."
            elif error.code in (401, 403):
                hint = " Check the API key."
            elif error.code == 429:
                hint = " Rate limited. Reduce --runs or try again."
            raise SpeedError(
                f"HTTP {error.code} from {url}:{hint} {detail[:400]}"
            ) from None
        except urllib.error.URLError as error:
            raise SpeedError(f"cannot connect to {url}: {error.reason}") from None


def round_ms(seconds):
    return round(seconds * 1000, 1) if seconds is not None else None


def round_value(value):
    return round(value, 1) if value is not None else None


def rate(tokens, seconds):
    if tokens is None or tokens <= 0 or seconds is None or seconds <= 0:
        return None
    return round_value(tokens / seconds)


def percentile(values, quantile):
    """Linear percentile, matching common latency percentile conventions."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def iter_sse(response, started):
    """Yield decoded SSE events at the time their terminating line arrives."""
    data = []
    done = False
    for raw in response:
        now = time.perf_counter()
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data), now - started
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
            if data[-1] == "[DONE]":
                done = True
    if data and not done:
        yield "\n".join(data), now - started


def percentile_ms(samples, key, quantile):
    values = sorted(sample[key] for sample in samples if sample.get(key) is not None)
    if not values:
        return None
    return round(percentile(values, quantile), 1)


def run_once(cfg, scenario, timeout=DEFAULT_TIMEOUT):
    """Run one request and return its measurements."""
    url = chat_url(cfg["base_url"])
    body = build_body(cfg["model"], scenario, cfg.get("reasoning_effort"))
    start = time.perf_counter()
    response = open_stream(url, body, cfg.get("api_key"), timeout)

    ttft = None
    first_content = None
    tool_ready = None
    tool_name = None
    finish_reason = None
    prompt_tokens = None
    completion_tokens = None
    reasoning_tokens = None
    cached_tokens = None
    cache_write_tokens = None
    cost_usd = None
    first_content_chunk = None
    last_content_chunk = None
    first_reasoning_chunk = None
    last_reasoning_chunk = None
    last_output_at = None
    output_events = 0
    chunk_gaps = []
    content_chunk_times = []
    content_chunks = 0
    reasoning_chunks = 0
    tool_chunks = 0
    arguments_by_call = {}

    try:
        with response:
            for payload, elapsed in iter_sse(response, start):
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError as error:
                    raise SpeedError(f"invalid stream event: {error}") from None
                if event.get("error"):
                    raise SpeedError(f"stream error: {json.dumps(event['error'])[:400]}")

                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get(
                        "completion_tokens", completion_tokens
                    )
                    details = usage.get("completion_tokens_details") or {}
                    reasoning_tokens = details.get(
                        "reasoning_tokens",
                        usage.get("reasoning_tokens", reasoning_tokens),
                    )
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = prompt_details.get("cached_tokens", cached_tokens)
                    cache_write_tokens = prompt_details.get(
                        "cache_write_tokens", cache_write_tokens
                    )
                    cost_usd = usage.get("cost", cost_usd)

                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}

                is_reasoning = any(
                    isinstance(delta.get(key), str) and delta.get(key)
                    for key in ("reasoning_content", "reasoning", "thinking")
                )
                content_text = delta.get("content")
                is_content = isinstance(content_text, str) and bool(content_text)
                tool_deltas = [
                    call
                    for call in (delta.get("tool_calls") or [])
                    if isinstance(call, dict)
                ]

                if finish_reason == "tool_calls" and tool_chunks:
                    tool_ready = elapsed
                if not (is_reasoning or is_content or tool_deltas):
                    continue
                if ttft is None:
                    ttft = elapsed
                if last_output_at is not None:
                    chunk_gaps.append(elapsed - last_output_at)
                last_output_at = elapsed
                output_events += 1

                if is_reasoning:
                    reasoning_chunks += 1
                    if first_reasoning_chunk is None:
                        first_reasoning_chunk = elapsed
                    last_reasoning_chunk = elapsed
                if is_content:
                    content_chunks += 1
                    if first_content_chunk is None:
                        first_content_chunk = elapsed
                    if content_text.strip():
                        content_chunk_times.append(elapsed)
                        if first_content is None:
                            first_content = elapsed
                    last_content_chunk = elapsed
                for call in tool_deltas:
                    tool_chunks += 1
                    function = call.get("function") or {}
                    name = (function.get("name") or "").strip()
                    if name:
                        tool_name = name
                    call_id = (
                        call["index"] if "index" in call else call.get("id", 0)
                    )
                    arguments = function.get("arguments") or ""
                    arguments_by_call[call_id] = (
                        arguments_by_call.get(call_id, "") + arguments
                    )
                if finish_reason == "tool_calls" and tool_ready is None:
                    tool_ready = elapsed
    except (OSError, http.client.HTTPException, TimeoutError) as error:
        elapsed = time.perf_counter() - start
        raise SpeedError(f"stream failed after {elapsed:.1f} s: {error}") from None

    end = time.perf_counter() - start
    if ttft is None:
        raise SpeedError(
            "no streamed tokens. Check that the endpoint supports stream=true "
            "and that the model name is correct."
        )
    if tool_chunks:
        if finish_reason != "tool_calls" or tool_ready is None:
            raise SpeedError("tool call is incomplete")
        for arguments in arguments_by_call.values():
            try:
                parsed_arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                raise SpeedError("tool-call arguments are incomplete") from None
            if not isinstance(parsed_arguments, dict):
                raise SpeedError("tool-call arguments must form an object")

    if completion_tokens is not None and reasoning_tokens is None and not reasoning_chunks:
        reasoning_tokens = 0
    content_tokens = None
    if completion_tokens is not None and reasoning_tokens is not None:
        reasoning_tokens = max(0, min(reasoning_tokens, completion_tokens))
        content_tokens = completion_tokens - reasoning_tokens

    decode_seconds = last_output_at - ttft
    prefill_tps = rate(prompt_tokens, ttft)
    output_tps = rate(
        completion_tokens - 1 if completion_tokens is not None else None,
        decode_seconds,
    )
    tpot_ms = (
        round(decode_seconds * 1000 / (completion_tokens - 1), 1)
        if completion_tokens and completion_tokens > 1 and decode_seconds > 0
        else None
    )
    reasoning_tps = rate(
        reasoning_tokens - 1
        if reasoning_tokens is not None and first_reasoning_chunk is not None
        else reasoning_tokens,
        last_reasoning_chunk - first_reasoning_chunk
        if first_reasoning_chunk is not None
        else None,
    )
    content_tps = rate(
        content_tokens - 1
        if content_tokens is not None and first_content_chunk is not None
        else content_tokens,
        last_content_chunk - first_content_chunk
        if first_content_chunk is not None
        else None,
    )
    if tool_chunks:
        content_tokens = None
        content_tps = None
    visible_gaps = [
        later - earlier
        for earlier, later in zip(content_chunk_times, content_chunk_times[1:])
    ]
    usable_times = [
        value for value in (first_content, tool_ready) if value is not None
    ]
    cache_hit_percent = (
        round(cached_tokens * 100 / prompt_tokens, 1)
        if cached_tokens is not None and prompt_tokens
        else None
    )

    return {
        "ttft_ms": round_ms(ttft),
        "first_content_ms": round_ms(first_content),
        "first_usable_ms": round_ms(min(usable_times) if usable_times else None),
        "tool_ready_ms": round_ms(tool_ready),
        "tool_name": tool_name,
        "e2e_ms": round_ms(end),
        "decode_ms": round_ms(decode_seconds),
        "inter_chunk_p50_ms": (
            round(percentile(chunk_gaps, 0.50) * 1000, 1) if chunk_gaps else None
        ),
        "inter_chunk_p95_ms": (
            round(percentile(chunk_gaps, 0.95) * 1000, 1) if chunk_gaps else None
        ),
        "inter_chunk_max_ms": round(max(chunk_gaps) * 1000, 1) if chunk_gaps else None,
        "visible_stall_p95_ms": (
            round(percentile(visible_gaps, 0.95) * 1000, 1) if visible_gaps else None
        ),
        "prefill_tps": prefill_tps,
        "output_tps": output_tps,
        "reasoning_tps": reasoning_tps,
        "content_tps": content_tps,
        "tpot_ms": tpot_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_hit_percent": cache_hit_percent,
        "cost_usd": cost_usd,
        "output_chunks": output_events,
        "content_chunks": content_chunks,
        "reasoning_chunks": reasoning_chunks,
        "tool_chunks": tool_chunks,
        "finish_reason": finish_reason,
    }


def aggregate(samples):
    """Return median numeric fields and labels shared by every sample."""
    median = {}
    for key in NUMERIC_KEYS:
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        if not values:
            median[key] = None
        elif key in TOKEN_KEYS:
            median[key] = int(round(statistics.median(values)))
        elif key == "cost_usd":
            median[key] = statistics.median(values)
        else:
            median[key] = round(statistics.median(values), 1)
    tool_names = {sample.get("tool_name") for sample in samples}
    finish_reasons = {sample.get("finish_reason") for sample in samples}
    median["tool_name"] = tool_names.pop() if len(tool_names) == 1 else None
    median["finish_reason"] = (
        finish_reasons.pop() if len(finish_reasons) == 1 else None
    )
    cache_values = [
        sample["cache_hit_percent"]
        for sample in samples
        if sample.get("cache_hit_percent") is not None
    ]
    median["cache_hit_percent"] = (
        round(statistics.median(cache_values), 1) if cache_values else None
    )
    return median


def fmt_ms(value):
    if value is None:
        return "n/a"
    if value >= 1000:
        return f"{value / 1000:,.2f} s"
    if value >= 10:
        return f"{value:,.0f} ms"
    return f"{value:.1f} ms"


def fmt_seconds(value_ms):
    if value_ms is None:
        return "n/a"
    return f"{value_ms / 1000:,.2f} s"


def fmt_tps(value):
    if value is None:
        return "n/a"
    number = f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"
    return number + " tok/s"


def metric(label, value, note=""):
    dots = "." * max(3, LABEL_WIDTH - len(label))
    line = f"  {label} {dots} {value}"
    return f"{line}  {note}" if note else line


def ttft_span(samples):
    values = [sample["ttft_ms"] for sample in samples if sample.get("ttft_ms") is not None]
    if len(values) < 2 or max(values) >= 1000:
        return ""
    return f"[{min(values):,.0f}-{max(values):,.0f} ms]"


def render(results, cfg):
    print(f"llmspeed {VERSION} · {cfg['model']}")
    print(f"{cfg['base_url']} · {cfg['runs']} run(s) per scenario")
    for result in results:
        print()
        samples = result.get("samples") or []
        title = result["name"]
        if len(samples) > 1:
            title += f" · median of {len(samples)}"
        print(title)
        if not samples:
            print(f"  error: {result.get('error', 'unknown error')}")
            continue
        if result.get("error"):
            print(f"  warning: {result['error']}")
        med = result["median"]

        print(metric("TTFT", fmt_ms(med["ttft_ms"]), ttft_span(samples)))
        if len(samples) > 1:
            print(metric("TTFT p95", fmt_ms(result["p95"]["ttft_ms"])))
        print(metric("first content", fmt_ms(med["first_content_ms"])))
        print(metric("first usable", fmt_ms(med["first_usable_ms"])))
        if med["tool_ready_ms"] is not None:
            note = f"({med['tool_name']})" if med.get("tool_name") else ""
            print(metric("tool call ready", fmt_seconds(med["tool_ready_ms"]), note))
        if med["prefill_tps"] is not None:
            note = f"({int(med['prompt_tokens']):,} prompt tok)"
            print(metric("prefill", fmt_tps(med["prefill_tps"]), note))
        else:
            print(metric("prefill", "n/a"))
        if med["output_tps"] is not None:
            note = f"({int(med['completion_tokens']):,} tok)"
            print(metric("output", fmt_tps(med["output_tps"]), note))
        else:
            print(metric("output", "n/a"))
        if med["tpot_ms"] is not None:
            print(metric("TPOT", fmt_ms(med["tpot_ms"])))
        print(metric("decode time", fmt_ms(med["decode_ms"])))
        if med["inter_chunk_p95_ms"] is not None:
            print(metric("chunk gap p95", fmt_ms(med["inter_chunk_p95_ms"])))
            print(metric("max chunk gap", fmt_ms(med["inter_chunk_max_ms"])))
        if med["visible_stall_p95_ms"] is not None:
            print(metric("visible stall p95", fmt_ms(med["visible_stall_p95_ms"])))
        if med["reasoning_tokens"]:
            rate = fmt_tps(med["reasoning_tps"])
            print(metric("reasoning", f"{int(med['reasoning_tokens']):,} tok @ {rate}"))
        if med["reasoning_tokens"] and med["content_tps"] is not None:
            visible = int(med["completion_tokens"]) - int(med["reasoning_tokens"])
            print(
                metric(
                    "content",
                    fmt_tps(med["content_tps"]),
                    f"({visible:,} tok)",
                )
            )
        if med["cache_hit_percent"] is not None:
            print(metric("cache hit", f"{med['cache_hit_percent']:.1f}%"))
        if med["cost_usd"] is not None:
            print(metric("cost", f"${med['cost_usd']:.6f}"))
        print(metric("end-to-end", fmt_seconds(med["e2e_ms"])))
        if len(samples) > 1:
            print(metric("E2E p95", fmt_seconds(result["p95"]["e2e_ms"])))
    print()
    print("Effective rates include network delay and queue time.")


def load_config(path):
    candidate = Path(path) if path else Path("speedtest.json")
    if not path and not candidate.exists():
        return {}
    try:
        with candidate.open(encoding="utf-8") as handle:
            conf = json.load(handle)
    except FileNotFoundError:
        raise SpeedError(f"config file not found: {candidate}") from None
    except json.JSONDecodeError as error:
        raise SpeedError(f"config file is not valid JSON: {candidate}: {error}") from None
    if not isinstance(conf, dict):
        raise SpeedError("config file must contain a JSON object")
    return conf


def ask(text, default=""):
    suffix = f" [{default}]" if default else ""
    answer = input(f"{text}{suffix}: ").strip()
    return answer or default


def confirm(text, default=False):
    mark = "[Y/n]" if default else "[y/N]"
    answer = input(f"{text} {mark}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def interactive_setup(args, conf):
    """Ask for a provider, API key, and model. Return an updated config."""
    if args.provider:
        provider_name = args.provider
        preset = get_provider(provider_name)
    else:
        names = [*PROVIDERS, "custom"]
        print("Select a provider:")
        for index, name in enumerate(names, 1):
            item = get_provider(name)
            place = item["base_url"] or "enter your own URL"
            print(f"  {index}. {item['label']} ({place})")
        choice = ask("Provider", "2" if args.base_url else "1")
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            provider_name = names[int(choice) - 1]
        else:
            provider_name = choice
        preset = get_provider(provider_name)

    if provider_name == "custom":
        base_url = (
            args.base_url
            or conf.get("base_url")
            or ask("Base URL", "http://localhost:8000/v1")
        )
    else:
        base_url = args.base_url or preset["base_url"]

    if provider_name == "custom":
        key_env = args.api_key_env or conf.get("api_key_env") or CUSTOM_KEY_ENV
    else:
        key_env = (
            args.api_key_env
            or preset["api_key_env"]
            or conf.get("api_key_env")
            or CUSTOM_KEY_ENV
        )
    api_key = args.api_key
    if not api_key and os.environ.get(key_env):
        api_key = os.environ[key_env]
        print(f"Using {key_env} from the environment.")
    elif not api_key:
        label = f"{preset['label']} API key"
        if provider_name == "custom":
            label += " (blank for none)"
        api_key = getpass.getpass(f"{label}: ").strip()
        if not api_key and provider_name != "custom":
            raise SpeedError(f"no API key. Set {key_env} or enter a key.")

    model = args.model
    if not model:
        models = []
        try:
            models = list_models(base_url, api_key)
        except SpeedError as error:
            print(f"llmspeed: {error}", file=sys.stderr)
            models = preset.get("models", [])
            if models:
                print("Using the documented model list.")
        if models:
            print("Available models:")
            for index, model_id in enumerate(models, 1):
                print(f"  {index}. {model_id}")
            while not model:
                choice = ask("Model (number or id)")
                if not choice:
                    print("Select a model.")
                elif choice.isdigit() and 1 <= int(choice) <= len(models):
                    model = models[int(choice) - 1]
                else:
                    model = choice
        else:
            while not model:
                model = ask("Model id")
                if not model:
                    print("Enter a model id.")

    updated = dict(conf)
    updated.update(
        {
            "base_url": base_url,
            "model": model,
            "api_key_env": key_env,
            "api_key": api_key,
        }
    )

    target = Path(args.config) if args.config else Path("speedtest.json")
    if confirm(f"Save these settings to {target}?"):
        if target.exists() and not confirm(f"{target} exists. Overwrite?"):
            return updated
        data = {
            "base_url": base_url,
            "model": model,
            "api_key_env": key_env,
            "runs": args.runs
            if args.runs is not None
            else conf.get("runs", DEFAULT_RUNS),
            "timeout": args.timeout
            if args.timeout is not None
            else conf.get("timeout", DEFAULT_TIMEOUT),
        }
        if conf.get("scenarios"):
            data["scenarios"] = conf["scenarios"]
        try:
            target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            raise SpeedError(f"cannot write {target}: {error}") from None
        print(
            f"Wrote {target}. The API key is not saved. "
            f"Set {key_env} before the next run."
        )
    return updated


def build_config(args, conf):
    preset = get_provider(args.provider) if args.provider else None
    if args.provider == "custom" and not (
        args.base_url
        or conf.get("base_url")
        or os.environ.get("LLMSPEED_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    ):
        raise SpeedError("no base URL for the custom provider. Set --base-url.")
    base_url = (
        args.base_url
        or (preset["base_url"] if preset else None)
        or conf.get("base_url")
        or os.environ.get("LLMSPEED_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = args.model or conf.get("model")
    if not model:
        raise SpeedError(
            'no model. Set --model, add "model" to the config file, '
            "or run in an interactive terminal."
        )
    if args.provider == "custom":
        key_env = args.api_key_env or conf.get("api_key_env") or CUSTOM_KEY_ENV
    elif preset:
        key_env = args.api_key_env or preset["api_key_env"] or "OPENAI_API_KEY"
    else:
        key_env = args.api_key_env or conf.get("api_key_env") or "OPENAI_API_KEY"
    api_key = args.api_key or os.environ.get(key_env) or conf.get("api_key")
    runs = args.runs if args.runs is not None else conf.get("runs", DEFAULT_RUNS)
    timeout = (
        args.timeout if args.timeout is not None else conf.get("timeout", DEFAULT_TIMEOUT)
    )
    try:
        runs = int(runs)
        timeout = float(timeout)
    except (TypeError, ValueError):
        raise SpeedError("runs must be an integer and timeout must be a number") from None
    if runs < 1:
        raise SpeedError("runs must be 1 or more")
    if timeout <= 0:
        raise SpeedError("timeout must be more than 0 seconds")
    return {
        "base_url": str(base_url),
        "model": str(model),
        "api_key": api_key,
        "api_key_env": key_env,
        "reasoning_effort": args.reasoning_effort,
        "runs": runs,
        "timeout": timeout,
    }


def select_scenarios(conf, names):
    raw = conf["scenarios"] if "scenarios" in conf else BUILTIN_SCENARIOS
    if not isinstance(raw, list) or not raw:
        raise SpeedError("scenarios must be a non-empty list")
    scenarios = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("name") or "messages" not in item:
            raise SpeedError("each scenario needs a name and messages")
        scenarios.append(item)
    if names:
        known = {scenario["name"] for scenario in scenarios}
        missing = [name for name in names if name not in known]
        if missing:
            raise SpeedError(f"unknown scenario: {', '.join(missing)}")
        scenarios = [scenario for scenario in scenarios if scenario["name"] in names]
    return scenarios


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="llmspeed",
        description="Speed test for OpenAI-compatible LLM endpoints: TTFT, "
        "first content, tool-call readiness, prefill, output, and end-to-end time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
        "  llmspeed.py --provider coralbricks\n"
        "  llmspeed.py --provider coralbricks --model deepseek-v4.1-flash-fast "
        "--reasoning-effort high\n"
        "  llmspeed.py --base-url https://api.openai.com/v1 --model gpt-4.1-mini\n"
        "  llmspeed.py --config speedtest.json --runs 5\n"
        "  llmspeed.py --config speedtest.json --scenario chat --json\n",
    )
    parser.add_argument(
        "-c", "--config", help="JSON config file (default: ./speedtest.json if it exists)"
    )
    parser.add_argument("--base-url", help="API root, for example https://host/v1")
    parser.add_argument("--model", help="model name; skips the model prompt")
    parser.add_argument(
        "--provider", help="provider preset: coralbricks or custom"
    )
    parser.add_argument("--api-key", help="API key (prefer --api-key-env)")
    parser.add_argument(
        "--api-key-env", help="environment variable that holds the API key"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        metavar="NAME",
        help="run only this scenario (repeatable)",
    )
    parser.add_argument(
        "--reasoning-effort",
        metavar="VALUE",
        help="reasoning effort for every request, for example low, high, or max",
    )
    parser.add_argument("--runs", type=int, help="requests per scenario (default 3)")
    parser.add_argument("--timeout", type=float, help="request timeout in seconds")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        conf = load_config(args.config)
        if not (args.model or conf.get("model")):
            if args.json or not sys.stdin.isatty():
                raise SpeedError(
                    'no model. Set --model, add "model" to the config file, '
                    "or run in an interactive terminal."
                )
            try:
                conf = interactive_setup(args, conf)
            except EOFError:
                raise SpeedError(
                    "input ended before the setup was complete"
                ) from None
            except KeyboardInterrupt:
                raise SpeedError("setup stopped") from None
        cfg = build_config(args, conf)
        scenarios = select_scenarios(conf, args.scenario)
    except SpeedError as error:
        print(f"llmspeed: {error}", file=sys.stderr)
        return 2

    results = []
    for index, scenario in enumerate(scenarios, 1):
        if not args.json:
            print(
                f"llmspeed: [{index}/{len(scenarios)}] {scenario['name']}, "
                f"{cfg['runs']} run(s)",
                file=sys.stderr,
                flush=True,
            )
        samples = []
        error = None
        for _ in range(cfg["runs"]):
            try:
                samples.append(run_once(cfg, scenario, cfg["timeout"]))
            except SpeedError as exc:
                error = str(exc)
                break
        if samples:
            results.append(
                {
                    "name": scenario["name"],
                    "samples": samples,
                    "median": aggregate(samples),
                    "p95": {
                        "ttft_ms": percentile_ms(samples, "ttft_ms", 0.95),
                        "first_content_ms": percentile_ms(
                            samples, "first_content_ms", 0.95
                        ),
                        "first_usable_ms": percentile_ms(
                            samples, "first_usable_ms", 0.95
                        ),
                        "tool_ready_ms": percentile_ms(samples, "tool_ready_ms", 0.95),
                        "e2e_ms": percentile_ms(samples, "e2e_ms", 0.95),
                    },
                }
            )
            if error:
                results[-1]["error"] = error
        else:
            results.append({"name": scenario["name"], "error": error})

    payload = {
        "version": VERSION,
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "runs": cfg["runs"],
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        render(results, cfg)
    return 1 if any("error" in result for result in results) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nllmspeed: stopped", file=sys.stderr)
        sys.exit(130)
