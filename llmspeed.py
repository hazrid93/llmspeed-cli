#!/usr/bin/env python3
"""llmspeed - a speed test for OpenAI-compatible LLM endpoints.

It measures what a user or an agent harness waits for: time to first token,
first visible content, tool-call readiness, prefill rate, output rate, and
end-to-end time.
"""

from __future__ import annotations

import argparse
import http.client
import json
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

NUMERIC_KEYS = (
    "ttft_ms",
    "first_content_ms",
    "tool_ready_ms",
    "e2e_ms",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "prefill_tps",
    "output_tps",
    "reasoning_tps",
)
TOKEN_KEYS = {"prompt_tokens", "completion_tokens", "reasoning_tokens"}

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


def build_body(model, scenario):
    body = {key: value for key, value in scenario.items() if key != "name"}
    body["model"] = model
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


def run_once(cfg, scenario, timeout=DEFAULT_TIMEOUT):
    """Run one request and return its measurements."""
    url = chat_url(cfg["base_url"])
    body = build_body(cfg["model"], scenario)
    start = time.perf_counter()
    response = open_stream(url, body, cfg.get("api_key"), timeout)

    ttft = None
    first_content = None
    tool_ready = None
    last_tool_at = None
    tool_name = None
    finish_reason = None
    prompt_tokens = None
    completion_tokens = None
    reasoning_tokens = None
    content_chunks = 0
    reasoning_chunks = 0
    tool_chunks = 0
    tool_chars = 0

    try:
        with response:
            for raw in response:
                now = time.perf_counter()
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

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
                is_content = isinstance(delta.get("content"), str) and delta.get("content")
                tool_deltas = [
                    call
                    for call in (delta.get("tool_calls") or [])
                    if isinstance(call, dict)
                ]

                if finish_reason == "tool_calls" and tool_chunks:
                    tool_ready = now - start
                if not (is_reasoning or is_content or tool_deltas):
                    continue
                if ttft is None:
                    ttft = now - start

                if is_reasoning:
                    reasoning_chunks += 1
                if is_content:
                    content_chunks += 1
                    if first_content is None:
                        first_content = now - start
                for call in tool_deltas:
                    tool_chunks += 1
                    last_tool_at = now - start
                    function = call.get("function") or {}
                    name = (function.get("name") or "").strip()
                    if name:
                        tool_name = name
                    tool_chars += len(function.get("arguments") or "")
    except (OSError, http.client.HTTPException) as error:
        elapsed = time.perf_counter() - start
        raise SpeedError(f"stream failed after {elapsed:.1f} s: {error}") from None

    end = time.perf_counter() - start
    if ttft is None:
        raise SpeedError(
            "no streamed tokens. Check that the endpoint supports stream=true "
            "and that the model name is correct."
        )
    if tool_chunks and tool_ready is None:
        tool_ready = last_tool_at

    estimated = False
    if not completion_tokens:
        estimated = True
        tool_tokens = round(tool_chars / 4) if tool_chars else 0
        completion_tokens = (
            content_chunks + reasoning_chunks + max(tool_tokens, 1 if tool_chunks else 0)
        )
    if reasoning_tokens is None:
        if reasoning_chunks:
            reasoning_tokens = reasoning_chunks
            estimated = True
        else:
            reasoning_tokens = 0
    reasoning_tokens = max(0, min(reasoning_tokens, completion_tokens))

    prefill_tps = prompt_tokens / ttft if prompt_tokens and ttft > 0 else None
    output_tps = (
        completion_tokens / (end - ttft)
        if completion_tokens and end > ttft
        else None
    )
    visible_at = first_content if first_content is not None else tool_ready
    reasoning_tps = (
        reasoning_tokens / (visible_at - ttft)
        if reasoning_tokens and visible_at is not None and visible_at > ttft
        else None
    )

    return {
        "ttft_ms": round(ttft * 1000, 1),
        "first_content_ms": (
            round(first_content * 1000, 1) if first_content is not None else None
        ),
        "tool_ready_ms": (
            round(tool_ready * 1000, 1) if tool_ready is not None else None
        ),
        "tool_name": tool_name,
        "e2e_ms": round(end * 1000, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "prefill_tps": round(prefill_tps, 1) if prefill_tps is not None else None,
        "output_tps": round(output_tps, 1) if output_tps is not None else None,
        "reasoning_tps": round(reasoning_tps, 1) if reasoning_tps is not None else None,
        "estimated": estimated,
        "finish_reason": finish_reason,
    }


def aggregate(samples):
    """Median of each numeric field, plus the last tool name and finish reason."""
    median = {}
    for key in NUMERIC_KEYS:
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        if not values:
            median[key] = None
        elif key in TOKEN_KEYS:
            median[key] = int(round(statistics.median(values)))
        else:
            median[key] = round(statistics.median(values), 1)
    median["tool_name"] = next(
        (sample["tool_name"] for sample in samples if sample.get("tool_name")), None
    )
    median["estimated"] = any(sample.get("estimated") for sample in samples)
    median["finish_reason"] = samples[-1].get("finish_reason")
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


def fmt_tps(value, estimated=False):
    if value is None:
        return "n/a"
    number = f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"
    return ("~" if estimated else "") + number + " tok/s"


def metric(label, value, note=""):
    dots = "." * max(3, LABEL_WIDTH - len(label))
    line = f"  {label} {dots} {value}"
    return f"{line}  {note}" if note else line


def ttft_span(samples):
    values = [sample["ttft_ms"] for sample in samples if sample.get("ttft_ms") is not None]
    if len(values) < 2 or max(values) >= 1000:
        return ""
    return f"[{min(values):,.0f}-{max(values):,.0f} ms]"


def output_span(samples):
    values = [
        sample["output_tps"] for sample in samples if sample.get("output_tps") is not None
    ]
    if len(values) < 2:
        return ""
    return f"[{min(values):,.0f}-{max(values):,.0f} tok/s]"


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
        print(metric("first content", fmt_ms(med["first_content_ms"])))
        if med["tool_ready_ms"] is not None:
            note = f"({med['tool_name']})" if med.get("tool_name") else ""
            print(metric("tool call ready", fmt_seconds(med["tool_ready_ms"]), note))
        if med["prefill_tps"] is not None:
            note = f"({int(med['prompt_tokens']):,} prompt tok)"
            print(metric("prefill", fmt_tps(med["prefill_tps"], med["estimated"]), note))
        else:
            print(metric("prefill", "n/a"))
        if med["output_tps"] is not None:
            note = f"({int(med['completion_tokens']):,} tok)"
            print(metric("output", fmt_tps(med["output_tps"], med["estimated"]), note))
        else:
            print(metric("output", "n/a"))
        if med["reasoning_tokens"]:
            rate = fmt_tps(med["reasoning_tps"])
            print(metric("reasoning", f"{int(med['reasoning_tokens']):,} tok @ {rate}"))
        print(metric("end-to-end", fmt_seconds(med["e2e_ms"])))
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


def build_config(args, conf):
    base_url = (
        args.base_url
        or conf.get("base_url")
        or os.environ.get("LLMSPEED_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = args.model or conf.get("model")
    if not model:
        raise SpeedError('no model. Set --model or "model" in the config file.')
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
        "  llmspeed.py --base-url https://api.openai.com/v1 --model gpt-4.1-mini\n"
        "  llmspeed.py --config speedtest.json --runs 5\n"
        "  llmspeed.py --config speedtest.json --scenario chat --json\n",
    )
    parser.add_argument(
        "-c", "--config", help="JSON config file (default: ./speedtest.json if it exists)"
    )
    parser.add_argument("--base-url", help="API root, for example https://host/v1")
    parser.add_argument("--model", help="model name")
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
    parser.add_argument("--runs", type=int, help="requests per scenario (default 3)")
    parser.add_argument("--timeout", type=float, help="request timeout in seconds")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        conf = load_config(args.config)
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
