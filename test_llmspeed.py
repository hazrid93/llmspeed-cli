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

    def test_stream_options_retry(self):
        cfg = dict(self.cfg, model="retry")
        sample = llmspeed.run_once(cfg, llmspeed.BUILTIN_SCENARIOS[0], 10)
        self.assertEqual(sample["completion_tokens"], 10)
        self.assertFalse(sample["estimated"])

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
