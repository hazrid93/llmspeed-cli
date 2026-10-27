# llmspeed

A speed test for OpenAI-compatible LLM endpoints. It measures the full request, not only TTFT and tokens per second.

TTFT alone does not show interactive latency. A reasoning model can decode fast and still show the first useful token late. A tool call adds time before an agent can act. llmspeed reports reasoning, content, and tool-call time separately.

Use Python 3.10 or later. The tool uses only the Python standard library.

## Quick start

CoralBricks:

```bash
git clone https://github.com/hazrid93/llmspeed-cli.git
cd llmspeed-cli
export CORAL_API_KEY=cb_...
python3 llmspeed.py --provider coralbricks
```

The tool reads the model list from `GET /models` and asks you to select one. It never runs all models.

Any OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=sk-...
python3 llmspeed.py --base-url https://api.openai.com/v1 --model gpt-4.1-mini
```

Local endpoint:

```bash
python3 llmspeed.py --base-url http://localhost:8000/v1 --model my-model
```

## Help

```bash
python3 llmspeed.py --help
```

| Option | Meaning |
| --- | --- |
| `-h`, `--help` | Show options and examples. |
| `-c`, `--config FILE` | Read a JSON config file. |
| `--provider NAME` | Provider preset: `coralbricks` or `custom`. |
| `--base-url URL` | API root for any endpoint. |
| `--model NAME` | Model name. Skips the model prompt. |
| `--api-key KEY` | API key. Prefer an environment variable. |
| `--api-key-env NAME` | Environment variable that holds the API key. |
| `--reasoning-effort VALUE` | Reasoning effort for every selected scenario. |
| `--scenario NAME` | Run only this scenario. Repeatable. |
| `--runs N` | Requests per scenario. Default `3`. |
| `--timeout SECONDS` | Timeout for each request. Default `120`. |
| `--json` | Print JSON only. |
| `--version` | Show the tool version. |

Examples:

```bash
# Fixed model and high reasoning
python3 llmspeed.py --provider coralbricks \
  --model deepseek-v4.1-flash-fast \
  --scenario reasoning \
  --reasoning-effort high

# Five runs, machine-readable
python3 llmspeed.py --config speedtest.json --runs 5 --json > result.json
```

## Providers

| Provider | Endpoint | API key variable |
| --- | --- | --- |
| `coralbricks` | `https://inference.coralbricks.ai/v1` | `CORAL_API_KEY` |
| `custom` | your URL | `OPENAI_API_KEY` |

`--provider` sets the endpoint and the key variable. For `custom`, also set `--base-url`.

With no model and an interactive terminal, the tool asks for the provider, the API key, and a model. It can save the result to `speedtest.json`. The wizard does not write the API key to the file.

CoralBricks docs: https://www.coralbricks.ai/docs

CoralBricks models: `deepseek-v4.1-flash-fast` and `deepseek-v4.1-flash`. Reasoning effort defaults to `none`. Set `--reasoning-effort high` to measure reasoning.

## Configuration

```bash
cp speedtest.example.json speedtest.json
python3 llmspeed.py --config speedtest.json
```

`speedtest.json` is in `.gitignore`. Do not commit API keys. If the file exists in the current directory, the tool uses it automatically.

| Key | Default | Meaning |
| --- | --- | --- |
| `base_url` | `https://api.openai.com/v1` | API root. |
| `model` | none | Model name. Required for non-interactive use. |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable that holds the key. |
| `api_key` | none | Key in the file. Use only for local endpoints. |
| `runs` | `3` | Requests per scenario. |
| `timeout` | `120` | Timeout in seconds. |
| `scenarios` | built-in | Request definitions. |

Each scenario needs `name` and `messages`. Other keys go into the request body, for example `max_tokens`, `temperature`, `tools`, and `reasoning_effort`. Some reasoning models reject `max_tokens`. Use `max_completion_tokens` for these models.

The tool sets `stream` to `true` and requests usage with `stream_options`. If an endpoint rejects `stream_options`, it retries without it.

## Output

```text
llmspeed 0.1.0 · deepseek-v4.1-flash-fast
https://inference.coralbricks.ai/v1 · 3 run(s) per scenario

chat · median of 3
  TTFT ................ 1.39 s
  TTFT p95 ............ 1.58 s
  first content ....... 1.39 s
  first usable ........ 1.39 s
  prefill ............. 12.2 tok/s  (17 prompt tok)
  output .............. 298 tok/s  (206 tok)
  TPOT ................ 3.4 ms
  decode time ......... 687 ms
  chunk gap p95 ....... 29 ms
  max chunk gap ....... 166 ms
  visible stall p95 ... 29 ms
  cache hit ........... 0.0%
  cost ................ $0.000249
  end-to-end .......... 2.07 s
  E2E p95 ............. 2.32 s
```

This is one measured example, not a benchmark guarantee. Run at least 20 times for a useful p95.

Rates are effective. Network delay and queue time are part of TTFT.

Exit codes: `0` for success, `1` for a request error, and `2` for a configuration or setup error.

## Metrics

| Metric | Meaning |
| --- | --- |
| TTFT | Time from request start to the first content, reasoning, or tool-call event. |
| First content | Time to the first non-whitespace content. Tool-only responses show `n/a`. |
| First usable | The earlier of first content or a complete tool call. This is not task completion. |
| Tool call ready | Time to the finish event with complete tool-call arguments. |
| Prefill | Prompt tokens divided by TTFT. This effective rate includes network and queue time. |
| Output | `(completion tokens - 1) / (last output event - first output event)`. |
| TPOT | Decode time divided by `(completion tokens - 1)`. It excludes TTFT. |
| Reasoning | Reasoning tokens after the first, divided by the reasoning event span. |
| Content | Content tokens after the first, divided by the content event span. It is `n/a` for mixed content and tool calls. |
| Decode time | Time from the first output event to the last output event. |
| Chunk gap p50/p95/max | Percentiles and maximum of delays between output SSE events. |
| Visible stall p95 | p95 of delays between non-whitespace content events. |
| TTFT/E2E p95 | Tail latency across successful runs. Use at least 20 runs for a useful tail measurement. |
| Cache hit | Cached prompt tokens divided by total prompt tokens. |
| Cost | Request cost from `usage.cost`, when the provider reports it. |
| End-to-end | Time from request start to `[DONE]` or stream close. |

The output shows median run values and p95 TTFT/E2E. Cost is the median per request, not the total. Token rates require token counts from `usage`. Without `usage`, token counts and token rates show `n/a`. Chunk-gap metrics remain available.

A provider can put multiple tokens in one SSE event. Token rates and chunk gaps are effective client-side measurements, not engine-internal metrics. llmspeed does not execute tools. Repeated runs can warm the prompt cache.

## Tests

```bash
python3 -m unittest -v
```

The tests use a local mock server. They do not use a network or an API key.

## License

MIT
