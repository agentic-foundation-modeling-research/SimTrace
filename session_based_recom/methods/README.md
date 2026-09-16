# Recommendation methods

Use the following upstream repositories for the recommendation models. This
repository provides the data preparation and evaluation utilities; model
implementations are maintained separately upstream.

## Upstream sources

| Method | Inputs | Source and setup documentation |
| --- | --- | --- |
| NARM | Item IDs | [Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch](https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch#readme) |
| RESTC | Item IDs | [SUSTechBruce/RESTC-Source-code](https://github.com/SUSTechBruce/RESTC-Source-code#readme) |
| RAIN | Item IDs | [zengxy20/RAIN](https://github.com/zengxy20/RAIN#readme) |
| DIMO | Item IDs and text features | [Zhang-xiaokun/DIMO](https://github.com/Zhang-xiaokun/DIMO#readme) |
| MMSBR | Item IDs, text, images, price, and category | [Zhang-xiaokun/MMSBR](https://github.com/Zhang-xiaokun/MMSBR#readme) |

## External setup and reproduction

1. Obtain the selected model from its upstream repository in a separate working
   directory. Follow its README and dependency files to set up the environment.
   Record the commit SHA and dependency versions for your experiment.
2. From the SIMTRACE repository root, prepare the datasets:

   ```bash
   session_based_recom/scripts/prepare_data.sh
   session_based_recom/scripts/prepare_data_mm.sh  # for DIMO and MMSBR
   ```

3. Connect the model's data loader to the appropriate directories under
   `session_based_recom/data_processed/`:

   | Methods | Real training | Synthetic training | Held-out test |
   | --- | --- | --- | --- |
   | NARM, RESTC, RAIN | `real/` | `synth_matched/` | The shared `test.txt` in either directory |
   | DIMO | `real_dimo_paper/` | `synth_dimo_paper/` | The corresponding directory's aligned test split |
   | MMSBR | `real_mmsbr_paper/` | `synth_mmsbr_paper/` | The corresponding directory's aligned test split |

   See the [data format](../README.md#generated-data-format) and
   [experimental setup](../README.md#experimental-setup) for vocabulary,
   feature, and split details. Data-loader adaptation is required; these
   directories are not guaranteed to work with upstream defaults.
4. Train each arm using the upstream entry point, select checkpoints on
   validation HR@5, and evaluate on the held-out real sessions. Keep the same
   vocabulary across arms and preserve training-time graph information for
   graph-based models. Export per-example predictions in canonical test order
   using the format in [`preds_io.py`](../scripts/preds_io.py).
5. For outputs matching the log format in
   [`collect_results.py`](../scripts/collect_results.py), run:

   ```bash
   cd session_based_recom
   python3 scripts/collect_results.py --results results
   python3 scripts/significance.py --preds results/preds --json results/significance.json
   ```

Upstream repositories have their own training interfaces. A shared SIMTRACE
adapter is not provided, and `scripts/tstr.sh` does not launch training. The
steps above describe the integration required to follow this benchmark's
protocol; an upstream checkout alone does not reproduce the experiments.
The exact upstream revisions used in the original experiments were not recorded.

## Paper references

- **NARM:** Jing Li et al. [Neural Attentive Session-based Recommendation](https://arxiv.org/abs/1711.04725). CIKM, 2017.
- **RESTC:** Zhongwei Wan et al. [Spatio-Temporal Contrastive Learning Enhanced GNNs for Session-based Recommendation](https://arxiv.org/abs/2209.11461). arXiv:2209.11461, 2022.
- **RAIN:** Xinyi Zeng et al. [RAIN: Reconstructed-aware in-context enhancement with graph denoising for session-based recommendation](https://www.sciencedirect.com/science/article/pii/S0893608024009857). Neural Networks, 184, 107056, 2025.
- **DIMO:** Xiaokun Zhang et al. [Disentangling ID and Modality Effects for Session-based Recommendation](https://arxiv.org/abs/2404.12969). SIGIR, 2024.
- **MMSBR:** Xiaokun Zhang et al. [Beyond Co-Occurrence: Multi-Modal Session-Based Recommendation](https://doi.org/10.1109/TKDE.2023.3309995). IEEE Transactions on Knowledge and Data Engineering, 36(4), 1450–1462, 2024.
