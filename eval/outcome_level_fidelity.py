from collections import Counter

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, norm

from eval.models import NormalizedSession, SEMANTIC_ACTIONS


class OutcomeLevelFidelity:
    """Outcome-level fidelity evaluation.

    Compares session-level outcome distributions (checkout, browse-only,
    abandoned) between real and synthetic data using chi-square,
    two-proportion z-tests, and a Jensen-Shannon divergence over the
    outcome-rate distribution.
    """

    OUTCOME_LABELS = ["has_checkout", "browse_only", "abandoned"]

    def evaluate(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> dict:
        real_flags = self._extract_flags(real)
        synth_flags = self._extract_flags(synthetic)

        real_rates = self._rates(real_flags, len(real))
        synth_rates = self._rates(synth_flags, len(synthetic))

        chi2_result = self._chi_square_test(real, synthetic)
        z_tests = self._z_tests(real_flags, synth_flags, len(real), len(synthetic))
        per_session = self._per_session_outcome_scores(real, synthetic)

        return {
            "outcome_rates": {
                "real": real_rates,
                "synthetic": synth_rates,
            },
            "js_divergence": self._js_divergence(real_rates, synth_rates),
            "chi_square_test": chi2_result,
            "z_tests": z_tests,
            "per_session_scores": per_session,
        }

    @staticmethod
    def _js_divergence(
        real_rates: dict[str, float], synth_rates: dict[str, float]
    ) -> float:
        """Jensen-Shannon divergence between the real and synthetic outcome-rate
        distributions (renormalized to sum to 1)."""
        labels = list(real_rates.keys())
        p = np.array([real_rates[k] for k in labels], dtype=float)
        q = np.array([synth_rates[k] for k in labels], dtype=float)
        eps = 1e-12
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        return float(jensenshannon(p, q) ** 2)  # squared = actual JSD

    @staticmethod
    def _primary_outcome(session: NormalizedSession) -> str:
        """Classify a session into a single primary outcome label."""
        actions = set(session.actions)
        if "checkout" in actions:
            return "checkout"
        if "add" in actions:
            return "abandoned"
        return "browse_only"

    def _per_session_outcome_scores(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> dict:
        """Binary per-session score: 1 if primary outcomes match, 0 if not."""
        real_by_id = {s.session_id: s for s in real}
        result = {}
        for synth_session in synthetic:
            real_session = real_by_id.get(synth_session.session_id)
            if real_session is None:
                continue
            real_outcome = self._primary_outcome(real_session)
            synth_outcome = self._primary_outcome(synth_session)
            result[synth_session.session_id] = {
                "outcome_score": 1 if real_outcome == synth_outcome else 0,
                "real_outcome": real_outcome,
                "synth_outcome": synth_outcome,
            }
        return result

    @staticmethod
    def _classify_session(session: NormalizedSession) -> dict[str, bool]:
        actions = set(session.actions)
        has_checkout = "checkout" in actions
        has_add = "add" in actions

        browse_actions = {
            "detail",
            "explore-search",
            "explore-goto",
            "explore-back",
            "explore-stay",
            "terminate",
        }
        browse_only = actions.issubset(browse_actions)
        abandoned = has_add and not has_checkout

        return {
            "has_checkout": has_checkout,
            "has_add_to_cart": has_add,
            "browse_only": browse_only,
            "abandoned": abandoned,
        }

    def _extract_flags(self, sessions: list[NormalizedSession]) -> dict[str, int]:
        counts = {label: 0 for label in self.OUTCOME_LABELS}
        for s in sessions:
            flags = self._classify_session(s)
            for label in self.OUTCOME_LABELS:
                counts[label] += int(flags[label])
        return counts

    @staticmethod
    def _rates(flag_counts: dict[str, int], n: int) -> dict[str, float]:
        if n == 0:
            return {k: 0.0 for k in flag_counts}
        return {k: v / n for k, v in flag_counts.items()}

    def _chi_square_test(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> dict:
        """Chi-square test comparing full action type distributions."""
        real_counts = Counter()
        synth_counts = Counter()
        for s in real:
            real_counts.update(s.actions)
        for s in synthetic:
            synth_counts.update(s.actions)

        all_actions = SEMANTIC_ACTIONS
        real_row = [real_counts.get(a, 0) for a in all_actions]
        synth_row = [synth_counts.get(a, 0) for a in all_actions]

        # Filter columns where both are zero to avoid degenerate contingency tables
        observed = np.array([real_row, synth_row], dtype=float)
        col_sums = observed.sum(axis=0)
        nonzero_mask = col_sums > 0
        observed = observed[:, nonzero_mask]

        if observed.shape[1] < 2:
            return {
                "chi2": 0.0,
                "p_value": 1.0,
                "dof": 0,
                "note": "insufficient categories",
            }

        chi2, p_val, dof, _ = chi2_contingency(observed)
        return {
            "chi2": float(chi2),
            "p_value": float(p_val),
            "dof": int(dof),
        }

    def _z_tests(
        self,
        real_flags: dict[str, int],
        synth_flags: dict[str, int],
        n_real: int,
        n_synth: int,
    ) -> dict:
        """Two-proportion z-test for each binary outcome."""
        results = {}
        for label in self.OUTCOME_LABELS:
            results[label] = self._two_proportion_z_test(
                real_flags[label],
                n_real,
                synth_flags[label],
                n_synth,
            )
        return results

    @staticmethod
    def _two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> dict:
        """Compute z-statistic and p-value for H0: p1 = p2."""
        if n1 == 0 or n2 == 0:
            return {"z_stat": 0.0, "p_value": 1.0, "effect_size": 0.0}

        p1 = x1 / n1
        p2 = x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)

        if p_pool == 0 or p_pool == 1:
            return {"z_stat": 0.0, "p_value": 1.0, "effect_size": 0.0}

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se

        p_value = 2 * (1 - norm.cdf(abs(z)))

        return {
            "z_stat": float(z),
            "p_value": float(p_value),
            "effect_size": float(abs(p1 - p2)),
        }
