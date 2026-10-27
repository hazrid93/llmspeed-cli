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
| `--provider NAME` | Provider preset: `coralbricks` or `custom`. |
| `--base-url URL` | API root for any endpoint. |
| `--model NAME` | Model name. Skips the model prompt. |
| `--reasoning-effort VALUE` | Reasoning effort for every selected scenario. |
| `--scenario NAME` | Run only this scenario. Repeatable. |
| `--runs N` | Requests per scenario. Default `3`. |
| `--json` | Print JSON only. |
| `--config FILE` | Read a JSON config file. |

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
  TTFT ................ 1.23 s
  first content ....... 1.23 s
  prefill ............. 13.9 tok/s  (17 prompt tok)
  output .............. 228 tok/s  (199 tok)
  end-to-end .......... 2.22 s

tool-call · median of 3
  TTFT ................ 1.48 s
  first content ....... 1.48 s
  tool call ready ..... 1.56 s  (get_weather)
  prefill ............. 187 tok/s  (276 prompt tok)
  output .............. 710 tok/s  (52 tok)
  end-to-end .......... 1.56 s
```

Rates are effective. Network delay and queue time are part of TTFT.

## Metrics

| Metric | Meaning |
| --- | --- |
| TTFT | Time to the first token of any type. |
| First content | Time to the first visible token. |
| Tool call ready | Time to the complete tool call. |
| Prefill | Prompt tokens divided by TTFT. |
| Output | Completion tokens divided by the time from the first token to the end of the stream. |
| Reasoning | Reasoning tokens and their rate. |
| Content | Completion tokens minus reasoning tokens, divided by the time from first content to the end of the stream. |
| End-to-end | Time from the request to the last stream event. |

The output shows the median of the runs. Token counts come from `usage`. If the endpoint does not send `usage`, the tool estimates the count and marks the rate with `~`.

llmspeed does not execute tools. The tool-call value stops when the tool call is complete. Token counts can differ from provider dashboards.

## Tests

```bash
python3 -m unittest -v
```

The tests use a local mock server. They do not use a network or an API key.

## License

MIT
