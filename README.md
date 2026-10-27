# llmspeed

A speed test for LLM endpoints. It shows the time that a user waits. It also separates reasoning tokens, visible content, and tool calls.

TTFT and tokens per second do not show interactive latency. A reasoning model can decode fast and still show the first useful token late. A tool call adds more time before an agent can act. llmspeed reports the full request time and the important checkpoints.

## What it measures

llmspeed sends chat requests to an OpenAI-compatible endpoint and reads the event stream. It reports these values:

- **TTFT**: time to the first token of any type.
- **First content**: time to the first visible token. Reasoning tokens do not count.
- **Tool call ready**: time to the complete tool call. An agent can act at this time.
- **Prefill**: effective input rate. The value is prompt tokens divided by TTFT. The value includes network delay and queue time.
- **Output**: effective output rate. The value is output tokens divided by the time after TTFT. The value includes reasoning tokens and tool-call tokens.
- **Reasoning**: reasoning token count and rate, when the endpoint reports them.
- **End-to-end**: time from the request to the last stream event.

The tool shows the median of 3 runs by default.

## Requirements

Python 3.10 or later. The tool uses only the Python standard library.

## Quick start

```bash
git clone https://github.com/hazrid93/llmspeed-cli.git
cd llmspeed
export OPENAI_API_KEY=sk-...
python3 llmspeed.py --base-url https://api.openai.com/v1 --model gpt-4.1-mini
```

For a local endpoint:

```bash
python3 llmspeed.py --base-url http://localhost:8000/v1 --model my-model --runs 5
```

## Configuration

Copy the example file and edit it:

```bash
cp speedtest.example.json speedtest.json
python3 llmspeed.py --config speedtest.json
```

`speedtest.json` is in `.gitignore`. Do not commit API keys.

### Top-level keys

| Key | Default | Meaning |
| --- | --- | --- |
| `base_url` | `https://api.openai.com/v1` | API root. The tool adds `/chat/completions`. |
| `model` | required | Model name. |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable that holds the API key. |
| `api_key` | none | API key in the file. Use only for local endpoints. |
| `runs` | `3` | Number of requests for each scenario. |
| `timeout` | `120` | Timeout in seconds for each request. |
| `scenarios` | built-in | List of request definitions. |

The tool reads the API key from the command line first, then from the environment variable, then from the file.

### Scenario keys

Each scenario needs a `name` and `messages`. All other keys go into the request body. Examples: `max_tokens`, `temperature`, `top_p`, `reasoning_effort`, `tools`, `tool_choice`.

Some endpoints reject `max_tokens` for reasoning models. For these endpoints, use `max_completion_tokens`:

```json
{
  "name": "reasoning",
  "max_completion_tokens": 1024,
  "messages": [{"role": "user", "content": "..."}]
}
```

llmspeed sets `stream` to `true` and requests usage data with `stream_options`. If an endpoint rejects `stream_options`, the tool retries without it.

## Example output

```text
llmspeed 0.1.0 · gpt-4.1-mini
https://api.openai.com/v1 · 3 run(s) per scenario

chat · median of 3
  TTFT ................ 412 ms  [402-431 ms]
  first content ....... 418 ms
  prefill ............. 621 tok/s  (256 prompt tok)
  output .............. 94.2 tok/s  (256 tok)
  end-to-end .......... 3.13 s

tool-call · median of 3
  TTFT ................ 980 ms
  first content ....... n/a
  tool call ready ..... 1.55 s  (get_weather)
  prefill ............. 420 tok/s  (412 prompt tok)
  output .............. 49.2 tok/s  (31 tok)
  end-to-end .......... 1.61 s

Effective rates include network delay and queue time.
```

## JSON output

Use `--json` for CI and dashboards:

```bash
python3 llmspeed.py --config speedtest.json --json > result.json
```

The output has one entry for each scenario and each run. Exit codes: 0 for success, 1 for a request error, 2 for a configuration error.

## Metric details

- The TTFT timer starts before the tool sends the request. It stops at the first event with visible content, reasoning content, or tool-call data. A role-only event does not count.
- The output rate uses the time after TTFT. Thus the value includes inter-token delay and the final stream gap.
- The reasoning rate uses reasoning tokens divided by the reasoning time. The reasoning time starts at TTFT and stops at the first visible content or tool call.
- Token counts come from the `usage` object. If the endpoint does not send `usage`, the tool counts stream chunks and marks the value with `~`.
- Reasoning tokens come from `completion_tokens_details.reasoning_tokens`. Some gateways put `reasoning_tokens` at the top level of `usage`. The tool reads both.
- llmspeed does not execute tools. The tool-call value stops when the tool call is complete.
- Network delay and provider queue time are part of TTFT. Run the test near the endpoint.
- Prompt caching changes the prefill value. State the cache condition when you compare results.
- A cold request is slower. Use 3 or more runs and compare the median values.
- Token counts can differ from provider dashboards because tokenizers can change.

## Tests

```bash
python3 -m unittest -v
```

The tests start a local mock server. They do not use a network or an API key.

## License

MIT
