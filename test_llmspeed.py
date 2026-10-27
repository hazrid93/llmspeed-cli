#!/usr/bin/env python3
"""Tests for llmspeed. They start a local mock SSE server and use no network."""

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import llmspeed


class FakeResponse:
    def __init__(self, events):
        self.lines = [
            line.encode()
            for event in [*events, "[DONE]"]
            for line in (
                f"data: {json.dumps(event) if isinstance(event, dict) else event}\n",
                "\n",
            )
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter(self.lines)


def metric_clock(events, stamps):
    """Timestamp each SSE frame and the final DONE and stream close."""
    assert len(events) == len(stamps)
    values = [0]
    for stamp in stamps:
        values.extend([stamp, stamp])
    values.extend([stamps[-1] + 0.1, stamps[-1] + 0.1, stamps[-1] + 0.2])
    return iter(values)


def event(delta=None, finish_reason=None, **extra):
    return {
        "choices": [
            {"delta": delta or {}, "finish_reason": finish_reason}
        ],
        **extra,
    }


class MockStream(BaseHTTPRequestHandler):
    """Stream a reasoning answer, or a tool call if the request has tools."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer test-key":
            payload = b'{"error": {"message": "invalid api key"}}'
            self.send_response(401)
        else:
            payload = json.dumps(
                {"object": "list", "data": [{"id": "mock-a"}, {"id": "mock-b"}]}
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get("model") == "retry" and "stream_options" in body:
            payload = b'{"error": {"message": "stream_options is not supported"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        if "tools" in body:
            events = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "get_weather", "arguments": ""},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": '{"city": "Kuala Lumpur"}'
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 90, "completion_tokens": 18},
                },
            ]
        else:
            events = [
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "Think. "},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "Answer. "},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {"delta": {"content": "The "}, "finish_reason": None}
                    ]
                },
                {
                    "choices": [
                        {"delta": {"content": "answer."}, "finish_reason": None}
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 64,
                        "completion_tokens": 10,
                        "completion_tokens_details": {"reasoning_tokens": 6},
                    },
                },
            ]

        if body.get("model") == "no-usage":
            events = [
                event
                for event in events
                if not (event.get("choices") == [] and "usage" in event)
            ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class LlmspeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockStream)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{port}/v1"
        cls.cfg = {
            "base_url": cls.base_url,
            "model": "mock",
            "api_key": None,
            "runs": 1,
            "timeout": 10,
        }

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_content_stream_splits_reasoning(self):
        sample = llmspeed.run_once(self.cfg, llmspeed.BUILTIN_SCENARIOS[0], 10)
        self.assertEqual(sample["prompt_tokens"], 64)
        self.assertEqual(sample["completion_tokens"], 10)
        self.assertEqual(sample["reasoning_tokens"], 6)
        self.assertGreater(sample["first_content_ms"], sample["ttft_ms"])
        self.assertGreater(sample["output_tps"], 0)
        self.assertGreater(sample["reasoning_tps"], 0)
        self.assertGreater(sample["content_tps"], 0)
        self.assertEqual(sample["finish_reason"], "stop")

    def test_exact_metric_boundaries(self):
        events = [
            event({"role": "assistant"}),
            event({"reasoning_content": "Think"}),
            event({"reasoning_content": " more"}),
            event({"content": "Answer"}),
            event({"content": " done"}),
            event(finish_reason="stop"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 9,
                    "prompt_tokens_details": {
                        "cached_tokens": 25,
                        "cache_write_tokens": 75,
                    },
                    "completion_tokens_details": {"reasoning_tokens": 4},
                    "cost": 0.001234,
                },
            },
        ]
        with mock.patch(
            "llmspeed.open_stream", return_value=FakeResponse(events)
        ), mock.patch(
            "llmspeed.time.perf_counter",
            side_effect=metric_clock(events, [0.1, 0.2, 0.3, 0.6, 0.7, 0.95, 1.0]),
        ):
            sample = llmspeed.run_once(
                {"base_url": "http://unused/v1", "model": "mock"},
                {"name": "exact", "messages": []},
            )
        expected = {
            "ttft_ms": 200,
            "first_content_ms": 600,
            "first_usable_ms": 600,
            "visible_stall_p95_ms": 100,
            "e2e_ms": 1200,
            "decode_ms": 500,
            "prefill_tps": 500,
            "output_tps": 16,
            "tpot_ms": 62.5,
            "reasoning_tps": 30,
            "content_tps": 40,
            "inter_chunk_p50_ms": 100,
            "inter_chunk_p95_ms": 280,
            "inter_chunk_max_ms": 300,
            "cache_hit_percent": 25,
            "cached_tokens": 25,
            "cache_write_tokens": 75,
            "cost_usd": 0.001234,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(sample[key], value, msg=key)

    def test_tool_arguments_must_be_complete(self):
        events = [
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "f", "arguments": '{"x":'},
                        }
                    ]
                }
            ),
            event(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": " 1}"}}
                    ]
                }
            ),
            event(finish_reason="tool_calls"),
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4}},
        ]
        with mock.patch(
            "llmspeed.open_stream", return_value=FakeResponse(events)
        ), mock.patch(
            "llmspeed.time.perf_counter",
            side_effect=metric_clock(events, [0.1, 0.2, 0.3, 0.4]),
        ):
            sample = llmspeed.run_once(
                {"base_url": "http://unused/v1", "model": "mock"},
                {"name": "tool", "messages": []},
            )
        self.assertEqual(sample["ttft_ms"], 100)
        self.assertEqual(sample["tool_ready_ms"], 300)
        self.assertEqual(sample["output_tps"], 30)
        self.assertIsNone(sample["first_content_ms"])

        incomplete = events[:1] + [event(finish_reason="tool_calls")]
        with mock.patch(
            "llmspeed.open_stream", return_value=FakeResponse(incomplete)
        ), mock.patch(
            "llmspeed.time.perf_counter",
            side_effect=metric_clock(incomplete, [0.1, 0.2]),
        ), self.assertRaisesRegex(
            llmspeed.SpeedError, "incomplete"
        ):
            llmspeed.run_once(
                {"base_url": "http://unused/v1", "model": "mock"},
                {"name": "tool", "messages": []},
            )

    def test_multiline_sse_and_same_event_tool_finish(self):
        class MultilineResponse(FakeResponse):
            def __init__(self):
                payload = json.dumps(
                    event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "f", "arguments": '{"x":1}'},
                                }
                            ]
                        },
                        finish_reason="tool_calls",
                    )
                )
                self.lines = [
                    f"data: {line}\n".encode()
                    for line in payload.split(" ")
                ] + [b"\n", b"data: [DONE]\n", b"\n"]

        with mock.patch(
            "llmspeed.open_stream", return_value=MultilineResponse()
        ), mock.patch(
            "llmspeed.time.perf_counter",
            side_effect=[0] + [0.1] * 20,
        ):
            sample = llmspeed.run_once(
                {"base_url": "http://unused/v1", "model": "mock"},
                {"name": "tool", "messages": []},
            )
        self.assertEqual(sample["ttft_ms"], 100)
        self.assertEqual(sample["tool_ready_ms"], 100)
        self.assertEqual(sample["first_usable_ms"], 100)

    def test_truncated_tool_stream_is_an_error(self):
        response = FakeResponse(
            [
                event(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"x":'}}
                        ]
                    }
                ),
                event(finish_reason="length"),
            ]
        )
        with mock.patch("llmspeed.open_stream", return_value=response), mock.patch(
            "llmspeed.time.perf_counter", side_effect=[0] + [0.1] * 20
        ), self.assertRaisesRegex(llmspeed.SpeedError, "incomplete"):
            llmspeed.run_once(
                {"base_url": "http://unused/v1", "model": "mock"},
                {"name": "tool", "messages": []},
            )

    def test_mixed_tool_content_does_not_inflate_content_rate(self):
        events = [
            event({"content": "Calling a tool"}),
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ]
                }
            ),
            event(finish_reason="tool_calls"),
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 8}},
        ]
        with mock.patch(
            "llmspeed.open_stream", return_value=FakeResponse(events)
        ), mock.patch(
            "llmspeed.time.perf_counter",
            side_effect=metric_clock(events, [0.1, 0.2, 0.3, 0.4]),
        ):
            sample = llmspeed.run_once(
                {"base_url": "http://unused/v1", "model": "mock"},
                {"name": "mixed", "messages": []},
            )
        self.assertIsNone(sample["content_tokens"])
        self.assertIsNone(sample["content_tps"])
        self.assertGreater(sample["output_tps"], 0)

    def test_zero_usage_never_produces_negative_rates(self):
        self.assertIsNone(llmspeed.rate(-1, 1))
        self.assertIsNone(llmspeed.rate(0, 1))
        self.assertIsNone(llmspeed.rate(1, 0))

    def test_first_content_ignores_whitespace(self):
        events = [
            event({"content": "\n\n"}),
            event({"content": "Hello"}),
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
        ]
        with mock.patch(
            "llmspeed.open_stream", return_value=FakeResponse(events)
        ), mock.patch(
            "llmspeed.time.perf_counter",
            side_effect=metric_clock(events, [0.1, 0.3, 0.4]),
        ):
            sample = llmspeed.run_once(
                {"base_url": "http://unused/v1", "model": "mock"},
                {"name": "chat", "messages": []},
            )
        self.assertEqual(sample["ttft_ms"], 100)
        self.assertEqual(sample["first_content_ms"], 300)
        self.assertEqual(sample["output_tps"], 5)
        self.assertIsNone(sample["visible_stall_p95_ms"])

    def test_tail_latency_and_cost_aggregation(self):
        samples = [
            {
                "ttft_ms": value,
                "e2e_ms": value * 2,
                "cost_usd": cost,
                "cache_hit_percent": 50,
            }
            for value, cost in zip([100, 200, 300, 400], [0.0001, 0.0002, 0.0003, 0.0004])
        ]
        self.assertEqual(llmspeed.percentile_ms(samples, "ttft_ms", 0.95), 385)
        self.assertEqual(llmspeed.percentile_ms(samples, "e2e_ms", 0.95), 770)
        median = llmspeed.aggregate(samples)
        self.assertEqual(median["cost_usd"], 0.00025)
        self.assertEqual(median["cache_hit_percent"], 50)

    def test_missing_usage_disables_token_rates(self):
        cfg = dict(self.cfg, model="no-usage")
        sample = llmspeed.run_once(cfg, llmspeed.BUILTIN_SCENARIOS[0], 10)
        self.assertIsNone(sample["prompt_tokens"])
        self.assertIsNone(sample["completion_tokens"])
        self.assertIsNone(sample["reasoning_tokens"])
        self.assertIsNone(sample["prefill_tps"])
        self.assertIsNone(sample["output_tps"])
        self.assertIsNone(sample["reasoning_tps"])
        self.assertIsNone(sample["content_tps"])
        self.assertIsNone(sample["tpot_ms"])
        self.assertEqual(sample["output_chunks"], 4)
        self.assertIsNotNone(sample["inter_chunk_p95_ms"])

    def test_tool_call_ready(self):
        sample = llmspeed.run_once(self.cfg, llmspeed.BUILTIN_SCENARIOS[2], 10)
        self.assertEqual(sample["tool_name"], "get_weather")
        self.assertIsNotNone(sample["tool_ready_ms"])
        self.assertGreaterEqual(sample["tool_ready_ms"], sample["ttft_ms"])
        self.assertIsNone(sample["first_content_ms"])
        self.assertEqual(sample["finish_reason"], "tool_calls")

    def test_stream_options_retry(self):
        cfg = dict(self.cfg, model="retry")
        sample = llmspeed.run_once(cfg, llmspeed.BUILTIN_SCENARIOS[0], 10)
        self.assertEqual(sample["completion_tokens"], 10)

    def test_max_completion_tokens(self):
        body = llmspeed.build_body(
            "model",
            {"name": "x", "max_completion_tokens": 100, "messages": []},
        )
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["max_completion_tokens"], 100)

    def test_provider_preset(self):
        args = llmspeed.parse_args(["--provider", "coralbricks", "--model", "m"])
        cfg = llmspeed.build_config(args, {})
        self.assertEqual(cfg["base_url"], "https://inference.coralbricks.ai/v1")
        self.assertEqual(cfg["api_key_env"], "CORAL_API_KEY")

    def test_reasoning_effort_cli(self):
        args = llmspeed.parse_args(
            [
                "--provider",
                "coralbricks",
                "--model",
                "m",
                "--reasoning-effort",
                "high",
            ]
        )
        cfg = llmspeed.build_config(args, {})
        self.assertEqual(cfg["reasoning_effort"], "high")
        body = llmspeed.build_body("m", {"name": "x", "messages": []}, "high")
        self.assertEqual(body["reasoning_effort"], "high")
        override = llmspeed.build_body(
            "m", {"name": "x", "messages": [], "reasoning_effort": "low"}, "high"
        )
        self.assertEqual(override["reasoning_effort"], "high")

        plain = llmspeed.build_config(
            llmspeed.parse_args(["--model", "m"]), {}
        )
        self.assertIsNone(plain["reasoning_effort"])
        self.assertNotIn(
            "reasoning_effort", llmspeed.build_body("m", {"name": "x", "messages": []})
        )

    def test_unknown_provider(self):
        args = llmspeed.parse_args(["--provider", "nope", "--model", "m"])
        with self.assertRaises(llmspeed.SpeedError):
            llmspeed.build_config(args, {})

    def test_list_models(self):
        self.assertEqual(
            llmspeed.list_models(self.base_url, "test-key"), ["mock-a", "mock-b"]
        )

    def test_interactive_setup_saves_without_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "speedtest.json"
            args = llmspeed.parse_args(
                ["--provider", "custom", "--base-url", self.base_url, "-c", str(target)]
            )
            with mock.patch("builtins.input", side_effect=["", "2", "y"]), mock.patch(
                "llmspeed.getpass.getpass", return_value="test-key"
            ), mock.patch.dict(os.environ, {"LLMSPEED_API_KEY": "", "OPENAI_API_KEY": ""}), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(io.StringIO()):
                conf = llmspeed.interactive_setup(args, {})
            self.assertEqual(conf["model"], "mock-b")
            self.assertEqual(conf["api_key"], "test-key")
            saved = json.loads(target.read_text())
            self.assertNotIn("api_key", saved)
            self.assertEqual(saved["base_url"], self.base_url)
            self.assertEqual(saved["model"], "mock-b")
            self.assertEqual(saved["api_key_env"], "OPENAI_API_KEY")

    def test_wizard_keyboard_interrupt(self):
        class TTY(io.StringIO):
            def isatty(self):
                return True

        stderr = io.StringIO()
        with mock.patch("llmspeed.load_config", return_value={}), mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": ""}
        ), mock.patch(
            "llmspeed.getpass.getpass", side_effect=KeyboardInterrupt
        ), mock.patch.object(
            llmspeed.sys, "stdin", TTY()
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            stderr
        ):
            code = llmspeed.main(
                ["--provider", "custom", "--base-url", self.base_url]
            )
        self.assertEqual(code, 2)
        self.assertIn("setup stopped", stderr.getvalue())

    def test_no_model_non_interactive_exits_2(self):
        stderr = io.StringIO()
        with mock.patch("llmspeed.load_config", return_value={}), mock.patch.object(
            llmspeed.sys, "stdin", io.StringIO()
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            stderr
        ):
            code = llmspeed.main(["--provider", "coralbricks"])
        self.assertEqual(code, 2)
        self.assertIn("no model", stderr.getvalue())

    def test_wizard_eof_exits_2(self):
        class TTY(io.StringIO):
            def isatty(self):
                return True

        stderr = io.StringIO()
        with mock.patch("llmspeed.load_config", return_value={}), mock.patch.dict(
            os.environ, {"CORAL_API_KEY": ""}
        ), mock.patch(
            "llmspeed.getpass.getpass", side_effect=EOFError
        ), mock.patch.object(
            llmspeed.sys, "stdin", TTY()
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            stderr
        ):
            code = llmspeed.main(["--provider", "coralbricks"])
        self.assertEqual(code, 2)
        self.assertIn("input ended", stderr.getvalue())

    def test_model_prompt_eof_exits_2(self):
        class TTY(io.StringIO):
            def isatty(self):
                return True

        stderr = io.StringIO()
        with mock.patch("llmspeed.load_config", return_value={}), mock.patch.dict(
            os.environ, {"CORAL_API_KEY": "test-key"}
        ), mock.patch(
            "llmspeed.list_models", return_value=["mock-a", "mock-b"]
        ), mock.patch(
            "llmspeed.ask", side_effect=EOFError
        ), mock.patch.object(
            llmspeed.sys, "stdin", TTY()
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            stderr
        ):
            code = llmspeed.main(["--provider", "coralbricks"])
        self.assertEqual(code, 2)
        self.assertIn("input ended", stderr.getvalue())

    def test_help_shows_options(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(
            SystemExit
        ) as raised:
            llmspeed.parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--reasoning-effort", stdout.getvalue())

    def test_human_output_includes_new_metrics(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = llmspeed.main(
                [
                    "--base-url",
                    self.base_url,
                    "--model",
                    "mock",
                    "--runs",
                    "2",
                    "--scenario",
                    "chat",
                ]
            )
        self.assertEqual(code, 0)
        output = stdout.getvalue()
        for label in ("first usable", "TPOT", "decode time", "E2E p95"):
            self.assertIn(label, output)

    def test_json_cli(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = llmspeed.main(
                [
                    "--base-url",
                    self.base_url,
                    "--model",
                    "mock",
                    "--runs",
                    "1",
                    "--scenario",
                    "chat",
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["model"], "mock")
        sample = payload["results"][0]["samples"][0]
        self.assertEqual(sample["completion_tokens"], 10)
        self.assertEqual(payload["results"][0]["median"]["ttft_ms"], sample["ttft_ms"])
        self.assertEqual(
            payload["results"][0]["p95"]["e2e_ms"], sample["e2e_ms"]
        )
        self.assertEqual(
            sample["first_usable_ms"], sample["first_content_ms"]
        )


if __name__ == "__main__":
    unittest.main()
