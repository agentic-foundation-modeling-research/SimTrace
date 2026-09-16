# Data generation

This package turns enriched buyer sessions from `src/data_collection` into
synthetic browser trajectories. An LLM-driven buyer agent observes a live
website, chooses one browser action at a time, and records the action, page
state, screenshots, and LLM calls for fidelity evaluation and downstream
model training.

The pipeline is configured through [`conf/experiments.yaml`](../../conf/experiments.yaml)
and launched from the repository root with one command:

```bash
python -m src.data_gen.main.run_experiments --config conf/experiments.yaml
```

| Stage | Purpose | Main code | Output |
|---|---|---|---|
| **1. Configure experiments** | Select input sessions, website, agent mode, model, reasoning modules, and evaluation data | [`main/run_experiments.py`](main/run_experiments.py), [`conf/experiments.yaml`](../../conf/experiments.yaml), [`conf/base.yaml`](../../conf/base.yaml) | Effective configuration and selected sessions |
| **2. Generate sessions** | Run concurrent browser sessions, simplify the DOM, ask the agent for actions, and execute them | [`main/run.py`](main/run.py), [`main/experiment.py`](main/experiment.py), [`main/model.py`](main/model.py), [`agent/`](agent/), [`executor/`](executor/) | Per-session actions, observations, screenshots, DOM snapshots, and LLM traces |
| **3. Combine and evaluate** | Combine completed sessions, evaluate fidelity against real sessions, and aggregate repeated runs | [`main/experiment.py`](main/experiment.py), [`eval/run_eval.py`](../../eval/run_eval.py), [`eval/aggregate_runs.py`](../../eval/aggregate_runs.py) | Combined synthetic data and `eval_results.json` |

## 1. Prepare the inputs

Install the project dependencies and Playwright browser once:

```bash
pip install -r requirements.txt
playwright install chromium
```

Export the credentials required by the LLM provider configured in
`conf/experiments.yaml`. For an OpenAI-compatible endpoint, for example:

```bash
export OPENAI_API_KEY="your-api-key"
```

Run commands from the repository root because the default configuration uses
repository-relative paths.

### Session input

`run.input_filepath` must point to the `sessions.jsonl` produced by the data
collection pipeline. Every line represents one real session:

```json
{
  "session_id": "session-001",
  "trajectory": ["explore-search", "detail", "add", "checkout"],
  "intent": "I am looking for a lightweight travel backpack",
  "persona_id": "39",
  "persona": "A price-conscious shopper who compares several options..."
}
```

Required fields:

| Field | Required by | Description |
|---|---|---|
| `session_id` | all modes | Stable identifier used for progress tracking and resume behavior |
| `trajectory` | all modes | Ordered high-level actions from the corresponding real session; used directly by `traj_cond` and still required by the shared loader for `persona` |
| `intent` | all modes | Shopping goal given to the synthetic buyer |
| `persona` | `persona`; optional for `traj_cond` | Natural-language buyer profile |
| `persona_id` | optional | Persona identifier copied into run metadata |

For optional fidelity evaluation, configure the real artifacts produced by
data collection:

- `eval.real_data`: `sessions_raw.csv`;
- `eval.product_catalog`: `products.csv`; and
- `eval.llm`: model settings used by evaluation.

## 2. Configure experiments

[`conf/base.yaml`](../../conf/base.yaml) defines the browser, parser, timeout,
recording, and fallback model settings. [`conf/experiments.yaml`](../../conf/experiments.yaml)
defines the experiment sweep.

Configuration is merged in this order, with later values taking precedence:

```text
conf/base.yaml
└── top-level defaults in conf/experiments.yaml
    └── per-experiment overrides
        └── supported command-line evaluation overrides
```

At minimum, replace the placeholder paths, website URL, and LLM settings in
`conf/experiments.yaml`:

```yaml
output_dir: ./results

llm:
  provider: openai
  model: gpt-5
  base_url: https://api.openai.com/v1
  request_timeout: 120

run:
  input_filepath: data/sample/sessions.jsonl
  start_url: https://example.com
  sessions_index: 0-99
  concurrency: 4
  num_runs: 1

eval:
  enabled: true
  real_data: data/sample/sessions_raw.csv
  product_catalog: data/sample/products.csv

experiments:
  - name: trajectory-baseline
    simulation_config:
      agent_mode: traj_cond
      traj_cond:
        persona: false
        act: single
        plan: false
        verifier: "no"
        feedback: false
```

`sessions_index` accepts an inclusive range, individual indices, or a mixture:

```yaml
sessions_index: 0-99
sessions_index: 0,4,8
sessions_index: 0-9,20,25-29
```

Out-of-range indices are reported and skipped.

### Agent modes

| Mode | Inputs | Behavior |
|---|---|---|
| `persona` | persona + intent | Zero-shot buyer: one action-generation call per step, without trajectory-conditioned planning, feedback, or verification |
| `traj_cond` | trajectory + intent, optionally persona | Follows the real session's high-level trajectory and enables reasoning modules independently |

### `traj_cond` modules

| Setting | Values | Purpose |
|---|---|---|
| `persona` | `true`, `false` | Include the session persona in action, plan, and rethink prompts |
| `act` | `single`, `multi` | Generate one action directly or generate candidates and select one |
| `plan` | `true`, `false` | Build or update a plan before choosing the next action |
| `verifier` | `"no"`, `"pre"`, `"post"` | Disable verification, verify each proposed action, or verify the completed session |
| `feedback` | `true`, `false` | Reflect on the prior action and new page state before the next action |

An experiment's `name` is a display label. Runtime behavior is controlled by
`simulation_config.agent_mode` and its module settings.

### Browser environment

[`executor/env.py`](executor/env.py) launches Playwright and executes the
agent's structured actions. [`executor/parser/parser.js`](executor/parser/parser.js)
simplifies the visible DOM and returns:

- simplified HTML;
- valid clickable, input, hoverable, and select targets;
- a semantic-target-to-URL map; and
- open-tab information.

The agent can click, type, hover, select, clear fields, press keys, scroll,
navigate, manage tabs, or terminate the session. Set browser behavior in
`conf/base.yaml`, including `HEADLESS`, timeouts, persistent profile paths,
tracing, recording, and optional DOM-tag refresh.

For example:

```bash
HEADLESS=true python -m src.data_gen.main.run_experiments \
  --config conf/experiments.yaml
```

When `environment.browser.user_data_dir` is configured, the runner creates a
clean browser-profile copy for each concurrent worker. A bootstrap profile can
therefore preserve required login state while each generated session starts
from the same baseline.

## 3. Run the complete pipeline

Run generation and enabled fidelity evaluation:

```bash
python -m src.data_gen.main.run_experiments \
  --config conf/experiments.yaml
```

Run generation without evaluation:

```bash
python -m src.data_gen.main.run_experiments \
  --config conf/experiments.yaml \
  --no-run-eval
```

Resume incomplete experiments:

```bash
python -m src.data_gen.main.run_experiments \
  --config conf/experiments.yaml \
  --continue
```

Override evaluation inputs without editing the YAML:

```bash
python -m src.data_gen.main.run_experiments \
  --config conf/experiments.yaml \
  --real-data path/to/sessions_raw.csv \
  --product-catalog path/to/products.csv \
  --post-verify-threshold 0.8
```

Experiments run sequentially. Sessions within one experiment run concurrently
according to `run.concurrency`. `run.num_runs` repeats the same setting in
separate numbered directories and aggregates their evaluation results.

If an existing output directory is detected without `--continue`, the runner
prompts to continue or overwrite it. Continue mode reads the saved
`sessions_input.jsonl`, skips completed session IDs, and retries incomplete
sessions.

## Output layout

Results are grouped by agent mode, a configuration-derived slug, and run
number:

```text
results/
├── runs_index.jsonl
└── traj_cond/
    └── <model>__<module-settings>__n<count>/
        ├── eval_results.json
        └── 1/
            ├── config_used.yaml
            ├── sessions_input.jsonl
            ├── session_data_<timestamp>.json
            ├── eval_results.json
            └── runs/
                └── <timestamp>_<id>/
                    ├── basic_info.json
                    ├── action_trace.json
                    ├── observation_trace.jsonl
                    ├── session_data.json
                    ├── post_verify_result.json  # verifier: post only
                    ├── api_trace/
                    ├── observation_trace/
                    ├── screenshot/
                    ├── raw_html/
                    ├── simp_html/
                    └── axtree/
```

Failed session directories contain `error.txt` and are excluded from the
combined session file.

### Combined record schema

Each item in `session_data_<timestamp>.json` represents one executed browser
step:

```json
{
  "session_id": "session-001",
  "timestamp": "2026-09-01T12:34:56.789012",
  "synthetic_action": "{\"action\": \"click\", \"target\": \"product-1\", \"description\": \"Open the first product\"}",
  "clicked_url": "https://example.com/products/product-1",
  "url": "https://example.com/collections/backpacks",
  "llm_call": ["runs/<run-id>/api_trace/api_trace_1.json"],
  "screenshot_path": "runs/<run-id>/screenshot/screenshot_0.png",
  "dom_snapshot_raw": "runs/<run-id>/raw_html/raw_html_0.html",
  "dom_snapshot_simplified": "runs/<run-id>/simp_html/simp_html_0.html",
  "axtree_snapshot": "runs/<run-id>/axtree/axtree_0.json"
}
```

`runs_index.jsonl` records the model, effective module settings, input
configuration, output folder, timestamp, and elapsed time for every completed
experiment run.

## Package structure

```text
data_gen/
├── agent/                  # buyer agents, memory, LLM calls, and prompts
├── executor/               # Playwright environment and DOM parser
└── main/
    ├── run_experiments.py  # experiment-sweep CLI
    ├── run.py              # configuration, resume, retries, and run registry
    ├── experiment.py       # per-session execution and trace writing
    └── model.py            # policy orchestration for persona and traj_cond
```
