# Clickstream-Reproduction Verifier (`packages/shop_arena/src/shop_arena/gen/build/verifiers` + `final_eval`)

Status: **Implemented** · Version: **0.1**
Owners: ShopArena

> A new caller-owned **`clickstream_replay`** build-loop verifier that
> replays real user **clickstream** trajectories against the generated
> **twin** with Playwright and gates the task on *strict reproduction*
> of each session's actions. Opt-in (registered only when a clickstream
> CSV is supplied); plus an advisory cross-iteration rollup merged into
> `final_eval.json`.

---

## 1. Overview

`shop_arena.gen` generates **twin** SandboxShops from live storefronts.
In twin mode (`catalog_source="ingest"`) the catalog is ingested verbatim
from the seed's Shopify feeds, so the twin's product/collection handles
and URL paths match the live store exactly.

We also hold real user **clickstream** CSVs — ordered per-session actions
on the live store (`session_id`, `timestamp`, `semantic_action`
∈ {`detail`, `add`, `explore-stay`, `terminate`, …}, `url` (a path like
`/products/<handle>`), `product_handle`, `landing_url`, `store_id`, …).

The existing navigation verifiers only probe *synthetic* reachability:
`nav_coverage` asserts every collection handle is linked from the nav,
and `routes_200` boots a dev server and checks that a *sampled* set of
bucket routes returns HTTP 2xx. Neither exercises the *real* paths users
actually walked, nor the stateful `add` → `/cart` flow.

This spec adds one thing:

1. **`clickstream_replay` verifier.** For a deterministic sample of
   sessions, it maps each step's path onto the generated twin
   (dev-server base URL + path — never the live URL) and uses Playwright
   to check the concrete action is reachable: a `detail` step must land
   on a rendered PDP, an `add` step must add the line and have `/cart`
   reflect it, and generic navigation must return `< 400`. Any
   non-reproducible step FAILs the task; missing data (no CSV, or a
   `store_id` filter that excludes everything) is a PASS skip.

The verifier is **opt-in**: it registers only when a clickstream CSV is
resolved (`--clickstream <path>`, or the `<seed>/clickstream.csv`
convention) *and* Playwright is importable with a Chromium binary
present — otherwise the factory warns and omits it, exactly like
`visual_judge`. It gates `visual_fix` and `gen_navigation` by default.

---

## 2. Terminology

- **Clickstream / trajectory.** An ordered sequence of a single user's
  actions in one session on the live store, grouped by `session_id`.
- **Semantic action.** The high-level action token to reproduce:
  `detail` (view a PDP), `add` (add to cart), `explore-stay`
  (scroll/browse in place), `terminate` (session end), or any other
  action carrying a path (generic navigation).
- **Twin.** The generated SandboxShop in `catalog_source="ingest"` mode,
  whose handles/paths mirror the live store.
- **Strict reproduction.** A step is *reproducible* iff its concrete
  action succeeds on the twin (PDP renders, cart reflects the line,
  navigation returns `< 400`).
- **Skipped step.** An unknown action with no path — advisory, never
  gates the verdict.

---

## 3. Current Status

Implemented in `shop_arena.gen`:

- `_clickstream.py` — pure CSV parse + trajectory model + path→twin-URL
  mapping (`load_sessions`, `ClickstreamEvent`, `Session`,
  `SampledSessions`).
- `clickstream_replay.py` — `ClickstreamReplayVerifier`, the
  `TrajectoryReplayer` seam, and `PlaywrightTrajectoryReplayer`
  (sync Playwright, one browser, a fresh context per session).
- `final_eval/clickstream.py` — `build_clickstream_subtree`, the
  advisory cross-iteration rollup merged into `final_eval.json`.
- Config (`clickstream`, `clickstream_max_sessions`,
  `clickstream_store_id`, `resolve_clickstream_path`), CLI flags
  (`--clickstream`, `--clickstream-max-sessions`,
  `--clickstream-store-id`), loop-factory gating, and a
  `is_python_playwright_available()` probe.

---

## 4. Desired Status

No pending work for v0.1. The add-to-cart selector heuristics
(`_ADD_TO_CART_SELECTOR`) track the react-vite template's product form
and are the one brittle point — they are module constants so a template
change is a one-line edit. Follow-ups (not in scope): richer action
semantics (search-query replay, collection-filter state), and a
per-session screenshot artifact for failed steps.

---

## 5. Proposal

### 5.1 Trajectory model + URL mapping (pure)

`_clickstream.py` parses the CSV with `csv.DictReader`, groups rows by
`session_id`, orders each session's events by integer-millisecond
`timestamp` (ties broken by CSV row order), optionally filters by
`store_id`, and samples deterministically: sort session ids, take the
first `max_sessions`. URL mapping uses the `url` column path verbatim
against the twin base URL (`base_url.rstrip("/") + path`), falling back
to the path component of `landing_url`.

### 5.2 Reachability semantics

Executed in one browser, a fresh context per session (fresh cart):

- `detail` → `goto` the PDP; reachable iff status `< 400` **and** a
  product heading rendered (not the 404 boundary).
- `add` → ensure on the PDP, click the add-to-cart control (the variant
  form posting to `/cart` with `intent=add`), then `goto /cart` and
  assert the line appears.
- `explore-stay` → no-op; reachable iff a page is loaded.
- `terminate` → no-op; always reproducible.
- any other action with a path → generic navigation; reachable iff
  status `< 400`.
- unknown action, no path → *skipped* (advisory).

### 5.3 Verdict

Any non-skipped, non-reproducible step → **FAIL** (feedback lists
session id / step index / action / mapped url / reason). Every step
reproducible (or no sessions to replay) → **PASS**. A CSV that cannot be
read, an exhausted per-task retry budget, or a replay/dev-server crash →
**ADVISORY** so the loop proceeds. A missing storefront tree → **FAIL**.

### 5.4 Persistence + rollup

The harness dispatch layer already writes
`runs/build/iters/<iter_id>/checks/verifiers/clickstream_replay.json`
per applicable iteration, storing the verifier's `details` verbatim
(doable / not-doable counts, per-session breakdown, failures). The
post-loop `final_eval` step calls `build_clickstream_subtree`, which
tallies those files into an advisory `clickstream` subtree in
`<out_dir>/final_eval.json` (no separate report dir).

---

## 6. Execution Table

| Task | Description | Status |
| ---- | ----------- | ------ |
| T1 | `_clickstream.py`: CSV parse, session model, deterministic sampling, path→twin-URL mapping | Done |
| T2 | `clickstream_replay.py`: verifier, `TrajectoryReplayer` seam, `PlaywrightTrajectoryReplayer` | Done |
| T3 | `is_python_playwright_available()` probe in `_skills.py` | Done |
| T4 | Config fields + `resolve_clickstream_path`; CLI flags | Done |
| T5 | Loop factory gating (register iff CSV resolved + Playwright available) | Done |
| T6 | `final_eval/clickstream.py` rollup + `final_eval.json` merge | Done |
| T7 | Tests: parsing, verifier verdicts (stub seams), aggregation | Done |

---

## 7. Appendix

### 7.1 Consumed CSV columns

`session_id`, `timestamp`, `semantic_action`, `raw_action`, `url`,
`product_handle`, `store_id`, `landing_url`. Other columns present in the
analytics export (`user_id`, `product_hash`, `persona`,
`product_category`, `price_bucket`, `price`, `session_outcome_custom`,
`created_at`, `product_id`, `url_hash`) are ignored.

### 7.2 `clickstream` subtree shape

```jsonc
"clickstream": {
  "ran": true,                        // false when no CSV / verifier never ran
  "sessions_sampled": 15,
  "dropped": 0,
  "iterations": [
    {"iter_id": "exec-0007", "verdict": "fail",
     "steps_total": 42, "doable": 39, "not_doable": 3}
  ],
  "final": {"verdict": "pass", "steps_total": 42,
            "doable": 42, "not_doable": 0}
}
```
