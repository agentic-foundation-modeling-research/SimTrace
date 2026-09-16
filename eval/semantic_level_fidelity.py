import logging
import os
from collections import defaultdict

# Prevent OMP threading conflicts between FAISS and sentence-transformers on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from eval.models import NormalizedSession

logger = logging.getLogger(__name__)

_RATIONALE_PROMPT = (
    "Given the following e-commerce browsing session (action sequence and product context), "
    "write a 2-3 sentence description of the user's likely shopping intent and behavior pattern.\n\n"
    "Session actions: {actions}\n"
    "Products viewed: {products}\n\n"
    "Behavioral intent description:"
)

_MAX_RATIONALE_SESSIONS = 50


class SemanticLevelFidelity:
    """Semantic-level fidelity evaluation.

    Measures product coherence within sessions using SBERT embeddings
    and FAISS, then optionally generates LLM rationales to compare
    behavioral intent between real and synthetic sessions.
    """

    def __init__(self, llm_config=None):
        self._model = None
        self._faiss = None
        self._llm_config = llm_config

    def _load_sbert(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_faiss(self):
        if self._faiss is not None:
            return
        import faiss

        self._faiss = faiss

    def evaluate(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> dict:
        self._load_sbert()
        self._load_faiss()

        real_coherence_dict = self._session_coherence_dict(real)
        synth_coherence_dict = self._session_coherence_dict(synthetic)

        product_sim = self._cross_product_similarity(real, synthetic)
        product_sim_session, per_session_nn = self._cross_product_sim_session(
            real, synthetic
        )

        # Build per-session scores combining NN similarity and intra-product coherence diff
        per_session: dict[str, dict] = {}
        for sid, nn_sim in per_session_nn.items():
            per_session[sid] = {"product_similarity_score": nn_sim}

        for sid in real_coherence_dict:
            if sid not in synth_coherence_dict:
                continue
            diff = abs(real_coherence_dict[sid] - synth_coherence_dict[sid])
            intra_score = max(0.0, 1.0 - diff)
            per_session.setdefault(sid, {})["intra_prod_score"] = intra_score

        real_stats = self._coherence_stats(list(real_coherence_dict.values()))
        synth_stats = self._coherence_stats(list(synth_coherence_dict.values()))
        coherence_gap = abs(real_stats["mean"] - synth_stats["mean"])
        diversity_ratio = (
            synth_stats["n"] / real_stats["n"] - 1.0 if real_stats["n"] else 0.0
        )

        results = {
            "product_coherence": {
                "real": real_stats,
                "synthetic": synth_stats,
            },
            "product_coherence_gap": coherence_gap,
            "product_diversity_ratio": diversity_ratio,
            "cross_product_similarity": product_sim,
            "cross_product_similarity_per_session": product_sim_session,
            "per_session_scores": per_session,
        }

        rationale_results = self._rationale_comparison(real, synthetic)
        results["rationale_comparison"] = rationale_results

        return results

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning L2-normalized vectors."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embeddings = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)

    def _session_coherence_dict(
        self, sessions: list[NormalizedSession]
    ) -> dict[str, float]:
        """Intra-session product coherence: mean pairwise cosine similarity per session_id."""
        result: dict[str, float] = {}
        for s in sessions:
            texts = [t for t in s.product_texts if t.strip()]
            if len(texts) < 2:
                continue
            embeddings = self._embed_texts(texts)
            # Cosine similarity matrix (already L2-normalized, so dot product = cosine)
            sim_matrix = embeddings @ embeddings.T
            n = sim_matrix.shape[0]
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            pairwise_sims = sim_matrix[mask]
            result[s.session_id] = float(np.mean(pairwise_sims))
        return result

    @staticmethod
    def _coherence_stats(scores: list[float]) -> dict:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        arr = np.array(scores)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(scores),
        }

    def _cross_product_similarity(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> dict:
        """Compare product category distributions using FAISS nearest-neighbor search.

        Real clickstream data is anonymized (product_hash / url_hash) and no
        longer carries product slugs, so we compare on product_category — the
        shared semantic signal both parsers populate.
        """
        real_texts = self._collect_product_texts(real)
        synth_texts = self._collect_product_texts(synthetic)

        if not real_texts or not synth_texts:
            return {
                "mean_nn_similarity": 0.0,
                "note": "insufficient product categories",
            }

        real_emb = self._embed_texts(real_texts)
        synth_emb = self._embed_texts(synth_texts)

        dim = real_emb.shape[1]
        index = self._faiss.IndexFlatIP(dim)
        index.add(real_emb)

        nn_sims, _ = index.search(synth_emb, 1)

        return {
            "mean_nn_similarity": float(np.mean(nn_sims)),
            "n_real_categories": len(real_texts),
            "n_synthetic_categories": len(synth_texts),
        }

    def _cross_product_sim_session(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> tuple[dict, dict[str, float]]:
        """Per-session cross-product similarity: pair real and synthetic by session_id,
        compute FAISS nearest-neighbor similarity within each pair, then average.

        Returns (aggregate_dict, per_session_dict).
        """
        real_by_id = {s.session_id: s for s in real}
        per_session_sims: dict[str, float] = {}

        for synth_session in synthetic:
            real_session = real_by_id.get(synth_session.session_id)
            if real_session is None:
                continue

            real_texts = list(
                {t.strip() for t in real_session.product_texts if t.strip()}
            )
            synth_texts = list(
                {t.strip() for t in synth_session.product_texts if t.strip()}
            )

            if not real_texts or not synth_texts:
                continue

            real_emb = self._embed_texts(real_texts)
            synth_emb = self._embed_texts(synth_texts)

            dim = real_emb.shape[1]
            index = self._faiss.IndexFlatIP(dim)
            index.add(real_emb)

            nn_sims, _ = index.search(synth_emb, 1)
            per_session_sims[synth_session.session_id] = float(np.mean(nn_sims))

        if not per_session_sims:
            return (
                {
                    "mean_nn_similarity": 0.0,
                    "note": "no matched session pairs with product data",
                },
                {},
            )

        agg = {
            "mean_nn_similarity": float(np.mean(list(per_session_sims.values()))),
            "n_sessions_compared": len(per_session_sims),
        }
        return agg, per_session_sims

    @staticmethod
    def _collect_product_texts(sessions: list[NormalizedSession]) -> list[str]:
        texts = set()
        for s in sessions:
            for c in s.product_texts:
                c = c.strip() if c else ""
                if c:
                    texts.add(c)
        return list(texts)

    def _rationale_comparison(
        self,
        real: list[NormalizedSession],
        synthetic: list[NormalizedSession],
    ) -> dict:
        """Generate and compare behavioral rationales via OpenAI.

        Pairs real and synthetic sessions by session_id so we compare matched
        trajectories. If multiple synthetic sessions share a session_id (e.g.
        several agent runs of the same trajectory), they are aggregated into
        one rationale per real session_id.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"skipped": True, "reason": "OPENAI_API_KEY not set"}

        try:
            from openai import OpenAI
        except ImportError:
            return {"skipped": True, "reason": "openai package not installed"}

        base_url = (
            getattr(self._llm_config, "base_url", None) if self._llm_config else None
        )
        client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )
        model = (
            getattr(self._llm_config, "model", None) if self._llm_config else None
        ) or "gpt-4o-mini"

        real_by_id = {s.session_id: s for s in real}
        synth_groups = defaultdict(list)
        for s in synthetic:
            if s.session_id in real_by_id:
                synth_groups[s.session_id].append(s)

        paired_ids = list(synth_groups.keys())[:_MAX_RATIONALE_SESSIONS]
        if not paired_ids:
            return {
                "skipped": True,
                "reason": "no overlapping session_ids between real and synthetic",
            }

        real_paired = [real_by_id[sid] for sid in paired_ids]
        synth_paired = [
            self._merge_sessions(sid, synth_groups[sid]) for sid in paired_ids
        ]

        real_rationales = self._generate_rationales(client, real_paired, model)
        synth_rationales = self._generate_rationales(client, synth_paired, model)

        if len(real_rationales) != len(synth_rationales) or not real_rationales:
            return {
                "skipped": True,
                "reason": "rationale generation failed for one or more pairs",
            }

        real_emb = self._embed_texts(real_rationales)
        synth_emb = self._embed_texts(synth_rationales)

        paired_sims = [
            float(np.dot(real_emb[i], synth_emb[i])) for i in range(len(real_emb))
        ]

        real_centroid = real_emb.mean(axis=0)
        synth_centroid = synth_emb.mean(axis=0)
        centroid_sim = float(
            np.dot(real_centroid, synth_centroid)
            / (np.linalg.norm(real_centroid) * np.linalg.norm(synth_centroid) + 1e-12)
        )

        return {
            "skipped": False,
            "n_sessions": len(paired_sims),
            "n_unmatched_real": len(real) - len(paired_ids),
            "n_unmatched_synthetic": sum(
                1 for s in synthetic if s.session_id not in real_by_id
            ),
            "mean_paired_similarity": float(np.mean(paired_sims)),
            "std_paired_similarity": float(np.std(paired_sims)),
            "centroid_similarity": centroid_sim,
        }

    @staticmethod
    def _merge_sessions(
        session_id: str, sessions: list[NormalizedSession]
    ) -> NormalizedSession:
        """Concatenate multiple sessions sharing a session_id into one."""
        if len(sessions) == 1:
            return sessions[0]
        merged = NormalizedSession(session_id=session_id)
        for s in sessions:
            merged.actions.extend(s.actions)
            merged.product_categories.extend(s.product_categories)
            merged.product_texts.extend(s.product_texts)
            merged.timestamps.extend(s.timestamps)
            merged.urls.extend(s.urls)
            merged.search_queries.extend(s.search_queries)
        return merged

    def _generate_rationales(
        self, client, sessions: list[NormalizedSession], model: str = "gpt-4o-mini"
    ) -> list[str]:
        rationales = []
        for s in sessions:
            products = [t for t in s.product_texts if t.strip()]
            prompt = _RATIONALE_PROMPT.format(
                actions=", ".join(s.actions[:20]),
                products=", ".join(products[:10]) if products else "(none)",
            )
            try:
                # Newer OpenAI models (gpt-5+) reject `max_tokens` (require
                # `max_completion_tokens`) and only accept the default temperature,
                # so we use the broadly-compatible param set. The limit is generous
                # to leave room for reasoning-model token overhead before the
                # short rationale text.
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                rationales.append(resp.choices[0].message.content.strip())
            except Exception as e:
                logger.warning(
                    "Rationale generation failed for session %s: %s", s.session_id, e
                )
                rationales.append("")

        return [r for r in rationales if r]
