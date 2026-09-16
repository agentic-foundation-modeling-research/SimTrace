import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from eval.models import NormalizedSession
from eval.sequence_level_fidelity import SequenceLevelFidelity
from eval.outcome_level_fidelity import OutcomeLevelFidelity
from eval.semantic_level_fidelity import SemanticLevelFidelity

logger = logging.getLogger(__name__)


class EvaluationAggregator:
    """Orchestrates all three fidelity evaluations in parallel."""

    def __init__(self, llm_config=None):
        self.evaluators = {
            "sequence_level": SequenceLevelFidelity(),
            "outcome_level": OutcomeLevelFidelity(),
            "semantic_level": SemanticLevelFidelity(llm_config=llm_config),
        }

    def run(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> dict:
        logger.info(
            "Starting evaluation: %d real sessions, %d synthetic sessions",
            len(real),
            len(synthetic),
        )
        start_time = time.time()
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(evaluator.evaluate, real, synthetic): name
                for name, evaluator in self.evaluators.items()
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error("%s failed: %s", name, e, exc_info=True)
                    results[name] = {"error": str(e)}

        total_elapsed = time.time() - start_time
        results["metadata"] = {
            "n_real_sessions": len(real),
            "n_synthetic_sessions": len(synthetic),
            "total_elapsed_seconds": round(total_elapsed, 2),
        }

        results["per_session_scores"] = self._merge_per_session_scores(results)
        self._print_summary(results)
        return results

    @staticmethod
    def _merge_per_session_scores(results: dict) -> dict:
        """Collect per_session_scores from each evaluator and merge into one dict."""
        _SCORE_KEYS = [
            "trajectory_score",
            "outcome_score",
            "product_similarity_score",
            "intra_prod_score",
        ]

        merged: dict[str, dict] = {}
        for key in ("sequence_level", "outcome_level", "semantic_level"):
            evaluator_result = results.get(key, {})
            if "error" in evaluator_result:
                continue
            per_session = evaluator_result.pop("per_session_scores", {})
            for sid, scores in per_session.items():
                if sid not in merged:
                    merged[sid] = {}
                merged[sid].update(scores)

        for sid, scores in merged.items():
            available = [scores[k] for k in _SCORE_KEYS if k in scores]
            scores["aggregate_score"] = float(np.mean(available)) if available else 0.0

        return merged

    @staticmethod
    def _print_summary(results: dict) -> None:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        meta = results.get("metadata", {})
        print(f"Real sessions:      {meta.get('n_real_sessions', '?')}")
        print(f"Synthetic sessions: {meta.get('n_synthetic_sessions', '?')}")
        print(f"Total time:         {meta.get('total_elapsed_seconds', '?')}s")

        seq = results.get("sequence_level", {})
        if "error" not in seq and "session_length" in seq:
            sl = seq.get("session_length", {})
            real_sl = sl.get("real", {})
            synth_sl = sl.get("synthetic", {})
            print("\n--- Sequence Level ---")
            print(
                f"  Session length (real):      mean={real_sl.get('mean', 0):.1f}  "
                f"std={real_sl.get('std', 0):.1f}  "
                f"p25/p50/p75={real_sl.get('p25', 0):.0f}/{real_sl.get('p50', 0):.0f}/{real_sl.get('p75', 0):.0f}  "
                f"n={real_sl.get('n', 0)}"
            )
            print(
                f"  Session length (synthetic): mean={synth_sl.get('mean', 0):.1f}  "
                f"std={synth_sl.get('std', 0):.1f}  "
                f"p25/p50/p75={synth_sl.get('p25', 0):.0f}/{synth_sl.get('p50', 0):.0f}/{synth_sl.get('p75', 0):.0f}  "
                f"n={synth_sl.get('n', 0)}"
            )
            print(f"  Session length JS divergence: {sl.get('js_divergence', 0):.4f}")
            print(
                f"  Action freq JS divergence:    "
                f"{seq.get('action_frequency', {}).get('js_divergence', 0):.4f}"
            )
            tm = seq.get("transition_matrix", {})
            print(f"  Transition matrix L1:         {tm.get('l1_distance', 0):.4f}")
            print(f"  Transition matrix Frobenius:  {tm.get('frobenius_norm', 0):.4f}")
            print(
                f"  Trajectory edit distance:     avg={seq.get('trajectory_difference', {}).get('edit_distance_avg', 0):.4f}  "
                f"std={seq.get('trajectory_difference', {}).get('edit_distance_std', 0):.4f}"
            )

        out = results.get("outcome_level", {})
        if "error" not in out and "chi_square_test" in out:
            chi2 = out.get("chi_square_test", {})
            print("\n--- Outcome Level ---")
            print(f"  Outcome JS divergence: {out.get('js_divergence', 0):.4f}")
            print(
                f"  Chi-square stat: {chi2.get('chi2', 0):.4f}  "
                f"p-value: {chi2.get('p_value', 0):.4f}"
            )
            for label, zt in out.get("z_tests", {}).items():
                print(
                    f"  {label}: z={zt.get('z_stat', 0):.3f}  "
                    f"p={zt.get('p_value', 0):.4f}  "
                    f"effect={zt.get('effect_size', 0):.4f}"
                )

        sem = results.get("semantic_level", {})
        if "error" not in sem and "product_coherence" in sem:
            coh = sem.get("product_coherence", {})
            print("\n--- Semantic Level ---")
            print(
                f"  Real coherence:      "
                f"{coh.get('real', {}).get('mean', 0):.4f} "
                f"(+/- {coh.get('real', {}).get('std', 0):.4f})"
            )
            print(
                f"  Synthetic coherence: "
                f"{coh.get('synthetic', {}).get('mean', 0):.4f} "
                f"(+/- {coh.get('synthetic', {}).get('std', 0):.4f})"
            )
            print(f"  Product coherence gap:   {sem.get('product_coherence_gap', 0):.4f}")
            print(f"  Product diversity ratio: {sem.get('product_diversity_ratio', 0):.4f}")
            xps = sem.get("cross_product_similarity", {})
            print(f"  Cross-product NN sim: {xps.get('mean_nn_similarity', 0):.4f}")
            xps_session = sem.get("cross_product_similarity_per_session", {})
            print(
                f"  Cross-product NN sim (per session): {xps_session.get('mean_nn_similarity', 0):.4f}"
            )

            rat = sem.get("rationale_comparison", {})
            if rat.get("skipped"):
                print(f"  Rationale comparison: skipped ({rat.get('reason', '')})")
            else:
                print(
                    f"  Rationale paired sim: {rat.get('mean_paired_similarity', 0):.4f}"
                )
                print(
                    f"  Rationale centroid sim: {rat.get('centroid_similarity', 0):.4f}"
                )

        print("=" * 60)
