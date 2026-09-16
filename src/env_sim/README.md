# Simulated Environment Generation

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

Built on the
[original ShopGym repository](https://github.com/agentic-foundation-modeling-research/shop-gym).

This repository focuses on an autonomous pipeline for building simulated sandboxes that preserve both what users see and how they interact. It consists of three modules:

1. **Exploration** collects browser evidence from the original environment,
   including screenshots, storefront data, navigation structure, and observed
   interactions.
2. **Generation** consolidates that evidence into an intermediate specification
   and transforms it into a runnable sandbox, replacing sensitive content with
   synthetic alternatives where configured.
3. **Verification** compares the generated sandbox with the original environment
   and feeds actionable failures back into generation for repair.

The generation and verification modules iterate until the sandbox satisfies the
configured requirements or the generation and retry budgets are exhausted.

```text
Original storefront
        |
        v
  Exploration -----> evidence + intermediate specification
                              |
                              v
                         Generation <------------------+
                              |                        |
                              v                        |
                    Runnable sandbox                   |
                              |                        |
                              v                        |
                       Verification -------------------+
                  visual fidelity + action replay
```

## What we added

### Evidence-driven exploration

The exploration agent inspects the source storefront and records evidence for its
visual theme, page layout, navigation hierarchy, catalog, and supported
interactions. Reference screenshots are retained by page area—including the
homepage, navigation, collections, product pages, cart/search surfaces, and
information pages—so generation can be verified against what users actually saw.

Exploration also prefetches structured storefront information such as products,
collections, homepage content, hero assets, and navigation data. The resulting
shop manual and evidence bundle form the input to generation.

### Visual-fidelity verification

The `visual_fidelity` verifier renders the generated sandbox with Playwright and
compares its screenshots with the reference screenshots collected during
exploration. A vision-capable LLM evaluates:

- layout and page hierarchy;
- color and typography;
- component correspondence;
- content density; and
- user-facing language consistency.

The evaluator returns an overall fidelity score from 0 to 10, category scores,
concrete discrepancies, and repair instructions. A page passes only when its
overall score reaches the configured threshold, its language matches the
reference, and it has no critical issue. The default threshold is `7.0`.

When verification fails, the issues and repair instructions are passed to the next
generation iteration. After the configured retry budget is exhausted, the check
becomes advisory so the pipeline can terminate instead of retrying indefinitely.

### Action-replay verification

The `clickstream_replay` verifier checks whether trajectories observed in real
clickstreams remain executable in the simulated environment. It samples sessions,
maps their paths onto the generated sandbox, and replays each action with
Playwright.

Examples include:

- `detail`: the product path must return a non-error response and render a product
  page;
- `add`: Playwright must activate an add-to-cart control and confirm the item in
  the cart;
- `explore-stay`: a browsable page must remain loaded;
- `terminate`: the session ends successfully; and
- other actions with paths: navigation must return a non-error response.

A trajectory passes only when every non-skipped action is reproduced successfully.
Failure feedback identifies the session, action, mapped URL, and reason, and is
returned to the generation loop for repair.

Action replay is most meaningful in twin mode (`--catalog-source ingest`), where
product and collection handles are preserved. It is enabled when a clickstream is
provided explicitly with `--clickstream` or discovered as
`<shop-manual>/clickstream.csv`.

## Quick start

### Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 20 or newer
- [`pnpm`](https://pnpm.io/) 9 or newer
- a supported coding-agent CLI: [`pi`](https://pi.dev/) (default) or
  [Claude Code](https://docs.claude.com/claude-code/)

### 1. Install dependencies

```bash
uv sync
pnpm install
uv run playwright install chromium
pnpm --filter @shop-gym/shop-backend build
```

Copy the environment template and configure the model credentials used by your
agent runtime and visual verifier:

```bash
cp .env.example .env
```

Validate the browser and agent-skill setup:

```bash
uv run shop-doctor
```

### 2. Explore the original storefront

```bash
uv run shop-explore https://example-shop.com
```

The command prints the generated shop-manual path when it completes:

```text
outputs/shop_manuals/<domain>/<run_id>/
```

This directory contains the intermediate manual, structured capabilities,
prefetched storefront data, and browser evidence used by the verifiers.

### 3. Generate a sandbox

Generate a faithful twin with visual verification and clickstream replay:

```bash
uv run shop-gen outputs/shop_manuals/<domain>/<run_id> \
  --name mock_shop \
  --template react-vite \
  --catalog-source ingest \
  --judges all \
  --clickstream /path/to/sessions_raw.csv
```
Notice: clickstream is coming from the output of data_collection step

Useful controls:

```bash
# Change the visual pass threshold (default: 7.0).
--visual-judge-pass-threshold 8.0

# Change the per-task verifier retry budget (default: 3).
--visual-retry-budget 5

# Limit the number of replayed clickstream sessions.
--clickstream-max-sessions 20

# Reduce concurrent Chromium instances on constrained machines.
--final-eval-visual-max-concurrency 1
```

The generated environment is written to:

```text
outputs/shops/mock_shop/
├── data/               # sandbox storefront data
├── runs/build/         # generation iterations and verifier feedback
├── final_eval.json     # final visual and clickstream summaries
└── ...                 # generated storefront application and metadata
```

Re-running the same command with the same `--name` resumes from cached pipeline
state where possible.

### 4. Rewrite an ingested shop with synthetic content

`packages/syn_gen/syn_gen.py` runs the complete rewrite sequence: product
titles, product descriptions, product images, collection titles, then homepage
hero and banner images. It reads `collections.json` and `homepage.json` beside
the product input by default and writes the rewritten dataset to `--output-dir`.
Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` before running it. Text rewrites
use Anthropic directly, while product and homepage image generation uses
OpenAI's Images API directly.


```bash
uv run python packages/syn_gen/syn_gen.py \
      outputs/shops/mock_shop/data/products.json \
      --collections-file outputs/shops/mock_shop/data/collections.json \
      --homepage-file outputs/shops/mock_shop/data/homepage.json \
      --workers 20 \
      --output-dir outputs/shops/mock_shop/ \
      --image-mode copyright-preserve
```

### 5. Serve the sandbox

```bash
pnpm shop:host start mock_shop 4000
```

The storefront is then available at `http://localhost:4000`. The backend uses port
`5000`. To inspect or stop local shops:

```bash
pnpm shop:host list -a
pnpm shop:host stop mock_shop
```

## Development

```bash
# Python
uv run pytest
uv run ruff check .
uv run ruff format .
uv run pyright

# TypeScript
pnpm -r build
pnpm -r test
pnpm lint
```

## License

This repository is available under the [MIT License](LICENSE).
