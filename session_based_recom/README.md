# Train-Synthetic-Test-Real recommendation

This folder evaluates whether synthetic buyer sessions can replace real
sessions for training next-item recommenders. Every model is evaluated on the
same held-out real sessions.

## Methods

This repository provides preprocessing and result-analysis utilities. Obtain
model implementations from the [upstream sources](methods/README.md), which
also document setup and how to use the prepared data for evaluation.

Referenced methods:

- NARM, RESTC, and RAIN use item IDs.
- DIMO and MMSBR use their published paper feature settings only: DIMO uses
  title + vendor + product type; MMSBR uses title, image, price, and category.

Reported metrics are **MRR@5, HR@5, MRR@10, and HR@10**. HR is Recall for the
single next-item target. Checkpoints are selected on validation HR@5.

## Experimental setup

`conf/tstr_data.yaml` pairs each synthetic trajectory set with its source real
clickstream. Synthetic `/products/<handle>` URLs are mapped back to the
real `product_id` through `products.csv`, and items are namespaced as
`store_id:product_id`.

Synthetic sessions are conditioned from real session IDs. Those IDs form the
training pool; every other real session is held out for testing. The same
minimum-frequency filter is applied independently to each arm, and only session
IDs that survive in both are retained. Test sessions are restricted to products
observed in both training corpora. The output directories therefore have:

- the same training session IDs;
- one integer vocabulary;
- a byte-identical real `test.txt`;
- different training sequences; the ID-method synthetic split is additionally
  subsampled after prefix expansion to match the real example count.

The two conditions are:

- `real_to_real`: real training sequences → held-out real test;
- `synth_to_real`: synthetic training sequences → the same held-out real test.

For NARM, RESTC, and RAIN, `synth_to_real` uses a seeded, volume-matched subset
with exactly as many prefix-expanded examples as `real_to_real`. The subsampling
does not truncate or rewrite sessions. DIMO and MMSBR continue to use the full
synthetic session corpus; their multimodal paper setting is unchanged.

For DIMO and MMSBR, the real arm uses real catalog content and the synthetic arm
uses synthetic catalog content, matching what the simulated buyers saw. Product
IDs and images remain aligned across conditions.

## Prepare data

From the repository root:

```bash
session_based_recom/scripts/prepare_data.sh
session_based_recom/scripts/prepare_data_mm.sh
```

The first command emits:

```text
data_processed/real/
data_processed/synth/
data_processed/synth_matched/
```

`synth_matched` is the training directory for the ID-only methods.
`synth` remains the full synthetic source used to build the multimodal
directories.

The multimodal command additionally emits:

```text
data_processed/{real,synth}_dimo_paper/
data_processed/{real,synth}_mmsbr_paper/
```

Feature extraction is cached under `data_processed/mm_cache/`. Use
`--pseudo mirror` only for a no-download smoke test; the evaluation setting uses
CLIP pseudo-modalities.

Preprocessing writes `alignment_report.json` and `fidelity_report.json`. Check
the fidelity warnings before training. Large support, next-item-distribution, or
transition differences are generation-quality failures and cannot be repaired
honestly by recommender hyperparameter tuning.

## Training and inference

See the [upstream sources and setup instructions](methods/README.md#external-setup-and-reproduction)
for each recommendation method.

## Collect results

For existing compatible experiment logs and prediction files (this release
does not generate model outputs), run from the repository root:

```bash
cd session_based_recom
python3 scripts/collect_results.py --results results
```

The report contains:

- real→real and synth→real mean ± sample standard deviation over seeds;
- paired cluster-bootstrap deltas over held-out sessions;
- non-inferiority at the configured margin;
- zero-parameter popularity priors for both training corpora.

Popularity priors are essential context. If their gap is already large, a model
gap may reflect item support or frequency concentration rather than learned
sequence structure.

## Generated data format

Every ID-only directory contains:

- `train.txt`, `test.txt`: pickled `(prefix_sequences, labels)`;
- `all_train_seq.txt`: unexpanded training sessions for graph methods;
- `test_groups.txt`: test-example-to-session mapping for cluster bootstrap;
- `item_dict.json`, `num_node.txt`, and `stats.json`.

Multimodal directories add the paper-required feature matrices and metadata.
See `preprocess/preprocess_mm.py` for their exact contracts.

## Contact

For questions about reproducing the recommendation experiments, please contact
Yunan Lu at [yl4021@columbia.edu](mailto:yl4021@columbia.edu).
