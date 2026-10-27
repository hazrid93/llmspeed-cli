#!/usr/bin/env python3
"""Tests for llmspeed. They start a local mock SSE server and use no network."""

import contextlib
import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import llmspeed


class MockStream(BaseHTTPRequestHandler):
    """Stream a reasoning answer, or a tool call if the request has tools."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
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
        self.assertFalse(sample["estimated"])
        self.assertEqual(sample["finish_reason"], "stop")

    def test_estimate_without_usage(self):
        cfg = dict(self.cfg, model="no-usage")
        sample = llmspeed.run_once(cfg, llmspeed.BUILTIN_SCENARIOS[0], 10)
        self.assertTrue(sample["estimated"])
        self.assertIsNone(sample["prompt_tokens"])
        self.assertIsNone(sample["prefill_tps"])
        self.assertEqual(sample["completion_tokens"], 4)
        self.assertEqual(sample["reasoning_tokens"], 2)
        self.assertGreater(sample["output_tps"], 0)

    def test_tool_call_ready(self):
        sample = llmspeed.run_once(self.cfg, llmspeed.BUILTIN_SCENARIOS[2], 10)
        self.assertEqual(sample["tool_name"], "get_weather")
        self.assertIsNotNone(sample["tool_ready_ms"])
        self.assertGreaterEqual(sample["tool_ready_ms"], sample["ttft_ms"])
        self.assertIsNone(sample["first_content_ms"])
        self.assertEqual(sample["finish_reason"], "tool_calls")

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


if __name__ == "__main__":
    unittest.main()
