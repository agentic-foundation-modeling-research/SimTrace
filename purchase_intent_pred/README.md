# Purchase Prediction

## Experimental Setup

This experiment asks whether synthetic clickstream data can support downstream
purchase prediction. It follows the final-outcome prediction task in
[*Can LLM Agents Simulate Multi-Turn Human Behavior?*](https://arxiv.org/abs/2503.20749):
the model receives the ground-truth shopping history immediately before the
last step and generates whether the final action is purchase or termination.

We evaluate two conditions on one shared, held-out real test set:

- **TRTR:** train on real trajectories and test on real sessions.
- **TSTR:** train on paired synthetic trajectories and test on real sessions.

TRTR and TSTR contain the same session IDs and number of examples.

## Task and data definition

Each example is one session. Events are ordered chronologically and truncated
immediately before the first explicit `checkout` or `terminate` action. Neither
terminal action nor any later event is present in the input. Only an explicitly
observed checkout is labeled `purchase`; every other final outcome is
`terminate`.

The model input is the raw action sequence. Product identity, title, category,
query, and price are deliberately excluded to minimize real-to-synthetic
catalog and text distribution shift. A prompt has this form:

```text
System: You predict the final action of an online shopping session. Use only
the observed history. Return exactly one JSON object and no explanation:
{"action": "purchase"} or {"action": "terminate"}.

User: Observed session history (chronological JSON):
[{"action":"detail"},{"action":"add"},{"action":"explore-stay"}]
Predict the unobserved final action.
```

The supervised target and generated response must be exactly one of:

```json
{"action":"purchase"}
```

```json
{"action":"terminate"}
```

Malformed JSON, additional fields, explanations, and any other action are
invalid and count as incorrect predictions.

## Dataset construction

[`data_prep.py`](data_prep.py) merges every configured store in
`purchase_intent_pred/conf/tstr_data.yaml` into common real and synthetic pools.

The following controls are enforced:

1. TRTR and TSTR use exactly the same unique paired IDs.
2. All stored histories contain at least one action and no terminal action.
3. The 500-session real test set contains 45 purchases and 455 terminations,
   approximately a 10:1 terminate-to-purchase ratio.

Prepare the final data:

```bash
python -m purchase_intent_pred.data_prep \
  --config purchase_intent_pred/conf/tstr_data.yaml \
  --output_dir data/purchase_intent_pred \
  --test_size 500 \
  --seed 13
```

The command writes:

| File | Use |
|---|---|
| `trtr_train.jsonl` | Real-only training |
| `tstr_train.jsonl` | Synthetic-only training |
| `validation.jsonl` | Reserved real development sessions |
| `test.jsonl` | Shared imbalanced real evaluation set |
| `dataset_summary.json` | Counts, prevalence, and verifier audit |
| `split_session_ids.json` | Exact IDs for leakage and reproducibility checks |

The dataset in the paper use 739 paired training IDs after verifier
filtering. TRTR and TSTR therefore contain 739 examples each.

## Model and training

The final downstream model is
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), used in text-only,
non-thinking mode. [`qwen_experiment.py`](qwen_experiment.py) fine-tunes a
separate QLoRA adapter for every condition and seed. Loss is applied only to
the target JSON tokens; prompt and padding tokens use label `-100`.

Default optimization settings are:

| Component | Setting |
|---|---|
| Base model | Qwen3.5-4B |
| Adaptation | QLoRA over all linear layers |
| Quantization | 4-bit NF4, double quantization |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Context length | 2,048 tokens; oldest history truncated first |
| Epochs | 5 |
| Per-device batch size | 10 |
| Gradient accumulation | 16 |
| Learning rate | `1e-5` |
| Scheduler / warmup | Cosine / 3% |
| Weight decay | 0 |
| Optimizer | Paged 8-bit AdamW |
| Seeds | 13, 17, 23, 29, 31 |

Install the focused dependencies:

```bash
pip install -r purchase_intent_pred/requirements.txt
```

Run the complete experiment:

```bash
python -m purchase_intent_pred.qwen_experiment \
  --data_dir data/purchase_intent_pred \
  --output_dir ckpt/purchase_intent_pred \
  --conditions trtr tstr \
  --seeds 13 17 23 29 31
```

Training uses Hugging Face `Trainer`, which shuffles examples independently for
each training seed. Completed adapters are reused only when
`training_manifest.json` matches the model, training data fingerprint,
condition, seed, and optimization settings. A stale or unverifiable adapter
causes an error rather than silent reuse. Use `--overwrite_existing` to retrain
intentionally and `--eval_only` to evaluate compatible saved adapters.

## Inference and evaluation

Inference uses deterministic greedy generation (`do_sample=False`) with at
most 32 new tokens. It does not compare candidate logits, construct a purchase
probability, select a threshold, or use validation labels. Every generated
action is evaluated directly as correct or incorrect.

F1 for the minority `purchase` action is the primary metric. Per-seed output
also records precision, recall, accuracy, invalid-generation rate, predicted
purchase rate, and the confusion matrix for error analysis. The final table
reports mean F1 and sample standard deviation across at least five training
seeds. The runner additionally provides a 95% percentile-bootstrap interval
over seeds and paired condition differences.
