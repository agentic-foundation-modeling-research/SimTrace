import asyncio
import json
import logging
from abc import ABC

import numpy as np

from . import context
from .gpt import async_chat, embed_text, load_prompt

logger = logging.getLogger(__name__)


class Memory:
    memories: list["MemoryPiece"] = []
    embeddings: np.ndarray
    importance: np.ndarray
    timestamp: int

    def __init__(self, agent):
        self.memories = []
        self.embeddings = np.array([])
        self.importance = np.array([])
        self.timestamp = 0
        self.agent = agent
        self.memory_importance_prompt = load_prompt(
            "memory_importance", agent._prompt_dir
        )
        self.read_lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()

        # NEW: track counts
        self.added_since_last_update: int = 0
        self.add_count_history: list[int] = []

    async def add_memory_piece(self, memory_piece):
        # async with self.update_lock:
        memory_piece.timestamp = self.timestamp
        self.memories.append(memory_piece)
        # NEW: increment and log
        self.added_since_last_update += 1
        logger.debug(
            f"Memory added (+1). Queued since last update = {self.added_since_last_update}; "
            f"total pieces = {len(self.memories)}"
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["read_lock"]
        del state["write_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.read_lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()

    async def update(self):
        async with self.write_lock:
            if self.embeddings is not None and len(self.memories) == len(
                self.embeddings
            ):
                # nothing new to process
                return

            logger.info(
                "Updating memory: computing embeddings & importance for new pieces"
            )

            async def get_embeddings():
                # first get memories with no embeddings
                start_idx = len(self.embeddings) if self.embeddings is not None else 0
                memory_to_embed = self.memories[start_idx:]
                if not memory_to_embed:
                    return np.array([]), 0
                inputs = [m.content for m in memory_to_embed]
                embeds = await embed_text(
                    inputs,
                    provider=self.agent.llm_config.provider,
                    model_name=self.agent.llm_config.embed_model,
                )
                embeds = np.array(embeds)
                for i, m in enumerate(memory_to_embed):
                    m.embedding = embeds[i]
                return embeds, len(memory_to_embed)

            async def update_importance():
                # update importance for new items
                start_idx = (
                    len(self.importance)
                    if isinstance(self.importance, np.ndarray)
                    else 0
                )
                memory_to_update = self.memories[start_idx:]
                if not memory_to_update:
                    return np.array([]), 0
                requests = [
                    [
                        {"role": "system", "content": self.memory_importance_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "persona": self.agent.persona,
                                    "intent": self.agent.intent,
                                    "memory": m.content,
                                    "plan": self.agent.current_plan.content
                                    if self.agent.current_plan
                                    else None,
                                }
                            ),
                        },
                    ]
                    for m in memory_to_update
                ]
                responses = await asyncio.gather(
                    *[
                        async_chat(
                            r,
                            json_mode=True,
                            log=False,
                            provider=self.agent.llm_config.provider,
                            model_name=self.agent.llm_config.model,
                            enable_thinking=self.agent.llm_config.enable_thinking,
                        )
                        for r in requests
                    ]
                )
                new_importance = [json.loads(r)["score"] for r in responses]
                new_importance = np.array(new_importance) / 10
                for i, m in enumerate(memory_to_update):
                    m.importance = new_importance[i]
                return new_importance, len(memory_to_update)

            (embeds, n_embeds), (new_importance, n_imps) = await asyncio.gather(
                get_embeddings(), update_importance()
            )

            async with self.read_lock:
                # merge embeddings
                if self.embeddings.size == 0:
                    self.embeddings = embeds
                elif embeds.size > 0:
                    self.embeddings = np.concatenate([self.embeddings, embeds])

                # merge importance
                if (
                    not isinstance(self.importance, np.ndarray)
                    or self.importance.size == 0
                ):
                    self.importance = new_importance
                elif new_importance.size > 0:
                    self.importance = np.concatenate([self.importance, new_importance])

            # NEW: summary logging and bookkeeping
            processed = max(n_embeds, n_imps)
            self.add_count_history.append(processed)
            logger.info(
                f"Updated memory: +{processed} new piece(s) this cycle "
                f"(embeddings: +{n_embeds}, importance: +{n_imps}). "
                f"Totals — pieces={len(self.memories)}, "
                f"embedded={self.embeddings.shape[0] if self.embeddings.size else 0}, "
                f"with_importance={self.importance.shape[0] if isinstance(self.importance, np.ndarray) else 0}"
            )
            # reset the counter now that we've processed the batch
            self.added_since_last_update = 0

    async def retrieve(
        self,
        query,
        n=20,
        include_recent_observation=False,
        include_recent_action=False,
        include_recent_plan=False,
        include_recent_thought=False,
        trigger_update=True,
        kind_weight={},
    ):
        if trigger_update:
            await self.update()
        async with self.read_lock:
            results = []
            if include_recent_observation:
                # must include the most recent observation
                results += [
                    m
                    for m in self.memories
                    if m.kind == "observation" and m.timestamp >= self.timestamp - 3
                ]
            if include_recent_action:
                # must include the most recent action
                results += [
                    m
                    for m in self.memories
                    if m.kind == "action" and m.timestamp >= self.timestamp - 5
                ]
            if include_recent_plan:
                results += [
                    m
                    for m in self.memories
                    if m.kind == "plan" and m.timestamp >= self.timestamp - 5
                ]
            if include_recent_thought:
                results += [
                    m
                    for m in self.memories
                    if m.kind == "thought" and m.timestamp >= self.timestamp - 5
                ]
            # make a copy for read
            # embedding = self.embeddings.copy()
            # importance = self.importance.copy()
            # memories = [m for m in self.memories]
            if self.embeddings.size == 0:
                if context.api_call_manager.get():
                    context.api_call_manager.get().retrieve_result.append(
                        [m.__json__() for m in results]
                    )
                return results

            query_embedding = (
                await embed_text(
                    [query],
                    provider=self.agent.llm_config.provider,
                    model_name=self.agent.llm_config.embed_model,
                )
            )[0]
            similarities = np.dot(self.embeddings, query_embedding)
            recencies = np.array([m.timestamp - self.timestamp for m in self.memories])
            recencies = np.exp(recencies)
            kind_weights = np.array([kind_weight.get(m.kind, 1) for m in self.memories])
            # print(similarities.size, recencies.size, self.importance.size)
            smallest_size = min(similarities.size, recencies.size, self.importance.size)
            scores = (
                similarities[:smallest_size]
                + recencies[:smallest_size]
                + self.importance[:smallest_size]
            ) * kind_weights[:smallest_size]
            # scores = (similarities + recencies + self.importance) * kind_weights
            top_indices = np.argsort(-scores)[:n]
            results += [self.memories[i] for i in top_indices]
            if context.api_call_manager.get():
                context.api_call_manager.get().retrieve_result.append(
                    [m.__json__() for m in results]
                )
            return results


class MemoryPiece(ABC):
    kind: str
    content: str
    embedding: np.ndarray
    importance: float
    memory: Memory
    timestamp: int

    def __init__(self, content, memory):
        self.kind = self.__class__.__name__.lower()
        self.embedding = np.array([])  # Initialize with an empty numpy array
        self.importance = -1
        self.memory = memory
        self.content = content
        # self.memory.add_memory_piece(self)

    def __json__(self):
        return {
            "kind": self.kind,
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
        }


class Observation(MemoryPiece):
    original: str

    def __init__(self, content, memory, original):
        super().__init__(content, memory)
        self.original = original


class Reflection(MemoryPiece):
    def __init__(self, content, memory):
        super().__init__(content, memory)


class Wonder(MemoryPiece):
    def __init__(self, content, memory):
        super().__init__(content, memory)


class Plan(MemoryPiece):
    next_step: str

    def __init__(self, content, memory, next_step):
        super().__init__(content, memory)
        self.next_step = next_step


class Action(MemoryPiece):
    raw_action: dict

    def __init__(self, content, memory, raw_action):
        super().__init__(content, memory)
        self.raw_action = raw_action


class Thought(MemoryPiece):
    def __init__(self, content, memory):
        super().__init__(content, memory)
