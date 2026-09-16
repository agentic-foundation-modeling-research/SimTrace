import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon

from eval.models import NormalizedSession, SEMANTIC_ACTIONS


class SequenceLevelFidelity:
    """Sequence-level fidelity evaluation.

    Compares real vs synthetic sessions on action sequence structure:
    session lengths, action type frequencies, and transition matrices.
    """

    def __init__(self):
        self.action_index = {a: i for i, a in enumerate(SEMANTIC_ACTIONS)}
        self.n_actions = len(SEMANTIC_ACTIONS)

    def evaluate(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> dict:
        real_lengths = [s.length for s in real]
        synth_lengths = [s.length for s in synthetic]

        real_freq = self._action_frequencies(real)
        synth_freq = self._action_frequencies(synthetic)

        real_trans = self._transition_matrix(real)
        synth_trans = self._transition_matrix(synthetic)

        traj_diff_avg, traj_diff_std, per_session_traj = self._trajectory_comparison(
            real, synthetic
        )

        return {
            "session_length": {
                "real": self._descriptive_stats(real_lengths),
                "synthetic": self._descriptive_stats(synth_lengths),
                "js_divergence": self._js_divergence_lengths(
                    real_lengths, synth_lengths
                ),
            },
            "action_frequency": {
                "real": {
                    a: float(real_freq[i]) for i, a in enumerate(SEMANTIC_ACTIONS)
                },
                "synthetic": {
                    a: float(synth_freq[i]) for i, a in enumerate(SEMANTIC_ACTIONS)
                },
                "js_divergence": self._js_divergence_vectors(real_freq, synth_freq),
            },
            "transition_matrix": {
                "real": real_trans.tolist(),
                "synthetic": synth_trans.tolist(),
                "l1_distance": float(np.sum(np.abs(real_trans - synth_trans))),
                "frobenius_norm": float(np.linalg.norm(real_trans - synth_trans)),
            },
            "trajectory_difference": {
                "edit_distance_avg": traj_diff_avg,
                "edit_distance_std": traj_diff_std,
            },
            "action_labels": SEMANTIC_ACTIONS,
            "per_session_scores": {
                sid: {"trajectory_score": score}
                for sid, score in per_session_traj.items()
            },
        }

    def _trajectory_comparison(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> tuple[float, float]:
        """Compute mean and std Levenshtein distance between matched real/synthetic action sequences."""
        real_by_id = {s.session_id: s for s in real}
        synth_by_id = {s.session_id: s for s in synthetic}

        per_session: dict[str, float] = {}
        distances = []
        for sid in real_by_id:
            if sid not in synth_by_id:
                continue
            a = real_by_id[sid].actions
            b = synth_by_id[sid].actions
            denom = max(len(a), len(b))
            dist = self._levenshtein(a, b) / denom if denom > 0 else 0.0
            distances.append(dist)
            per_session[sid] = 1.0 - dist  # flip: 1=identical, 0=worst

        if not distances:
            return 0.0, 0.0, {}

        arr = np.array(distances, dtype=float)
        return float(np.mean(arr)), float(np.std(arr)), per_session

    @staticmethod
    def _levenshtein(a: list[str], b: list[str]) -> int:
        if len(a) < len(b):
            a, b = b, a
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev = dp[0]
            dp[0] = i
            for j, cb in enumerate(b, 1):
                temp = dp[j]
                dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[len(b)]

    @staticmethod
    def _descriptive_stats(values: list[int | float]) -> dict:
        if not values:
            return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "n": 0}
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "n": len(values),
        }

    def _action_frequencies(self, sessions: list[NormalizedSession]) -> np.ndarray:
        """Normalized frequency vector over SEMANTIC_ACTIONS."""
        counts = Counter()
        total = 0
        for s in sessions:
            for a in s.actions:
                counts[a] += 1
                total += 1

        freq = np.zeros(self.n_actions)
        for i, action in enumerate(SEMANTIC_ACTIONS):
            freq[i] = counts.get(action, 0)

        if total > 0:
            freq /= total
        return freq

    def _transition_matrix(self, sessions: list[NormalizedSession]) -> np.ndarray:
        """Row-normalized transition matrix P(action_j | action_i)."""
        counts = np.zeros((self.n_actions, self.n_actions))
        for s in sessions:
            for a_from, a_to in zip(s.actions[:-1], s.actions[1:]):
                i = self.action_index.get(a_from)
                j = self.action_index.get(a_to)
                if i is not None and j is not None:
                    counts[i, j] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return counts / row_sums

    def _js_divergence_lengths(
        self, real: list[int | float], synth: list[int | float]
    ) -> float:
        """JS divergence over session length distributions, binned by length."""
        if not real or not synth:
            return 1.0

        max_len = max(max(real), max(synth)) + 1
        real_hist = np.zeros(max_len)
        synth_hist = np.zeros(max_len)

        for v in real:
            real_hist[int(v)] += 1
        for v in synth:
            synth_hist[int(v)] += 1

        return self._js_divergence_vectors(
            real_hist / real_hist.sum(),
            synth_hist / synth_hist.sum(),
        )

    @staticmethod
    def _js_divergence_vectors(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence between two probability vectors."""
        eps = 1e-12
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        return float(jensenshannon(p, q) ** 2)  # squared = actual JSD
