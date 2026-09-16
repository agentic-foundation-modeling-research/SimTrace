import inspect
import json
import logging
import time
from typing import Optional

from . import context
from .constants import ACT_MULTI_COUNT, VERIFIER_PASS_THRESHOLD
from .gpt import async_chat, load_prompt
from .memory import (
    Action,
    Memory,
    MemoryPiece,
    Observation,
    Plan,
    Thought,
)

logger = logging.getLogger(__name__)


# context manager to log api calls
class LogApiCall:
    def __init__(self, agent: "UXAgent") -> None:
        self.agent = agent

    def __enter__(self):
        self.agent.api_call_count += 1
        self._count = (
            self.agent.api_call_count
        )  # capture before concurrent calls can increment further
        logger.info("API call count: %s", self._count)
        self.method_name = inspect.currentframe().f_back.f_code.co_name
        self.retrieve_result = []
        self.request = []
        self.response = []
        self.start_time = time.time()
        self.price_cost = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model_name = []
        context.api_call_manager.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        context.api_call_manager.set(None)
        elapsed = time.time() - self.start_time
        trace_data = {
            "method_name": self.method_name,
            "request": self.request,
            "response": self.response,
            "retrieve_result": self.retrieve_result,
            "time": elapsed,
            "price_cost": self.price_cost,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_name": self.model_name,
        }
        with open(
            context.run_path.get() / "api_trace" / f"api_trace_{self._count}.json",
            "w",
        ) as f:
            json.dump(trace_data, f)
        logger.info(f"API Call time: {elapsed}")


class UXAgent:
    deny_list = ["Crime", "crime", "Security", "security"]
    _prompt_dir = "ux_prompts"

    def __init__(self, persona: str, intent: str, llm_config=None) -> None:
        self.perceive_prompt = load_prompt("perceive", self._prompt_dir)
        self.planning_prompt = load_prompt("planning", self._prompt_dir)
        self.action_prompt = load_prompt("action", self._prompt_dir)
        self.feedback_prompt = load_prompt("feedback", self._prompt_dir)
        self.memory = Memory(self)
        self.persona = persona
        self.intent = intent
        self.llm_config = llm_config
        self.current_plan: Optional[Plan] = None
        self.api_call_count = 0
        self.last_reflect_index = 0
        self.observation: Optional[Observation] = None

    async def perceive(self, environment):
        environment_full = json.dumps(environment)

        for denied_word in self.deny_list:
            environment_full = environment_full.replace(denied_word, "***")
        # print(environment_full)
        logger.info("agent perceiving environment...")
        with LogApiCall(self):
            request = [
                {"role": "system", "content": self.perceive_prompt},
                {"role": "user", "content": environment_full},
            ]
            result = await async_chat(
                request,
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
                llm_config=self.llm_config,
            )
            result = json.loads(result)
            print(result)
            self.observation = result["observations"]
            await self.memory.add_memory_piece(
                Observation(result["observations"][0], self.memory, environment)
            )

    @staticmethod
    def format_memories(
        memories: list[MemoryPiece], sort_by_kind=True, importance=True
    ) -> list[str]:
        # sort by kind and timestamp
        if sort_by_kind:
            memories = sorted(memories, key=lambda x: (x.kind, x.timestamp))

        if not importance:
            memories_str = [
                f"""timestamp: {m.timestamp}; content: {m.content}""" for m in memories
            ]
            return memories_str

        importances_str = [
            f"{m.importance:.2f}" if m.importance != -1 else "N/A" for m in memories
        ]
        memories_str = [
            f"""timestamp: {m.timestamp}; kind: {m.kind}; importance: {i}, content: {m.content}"""
            for m, i in zip(memories, importances_str)
        ]
        return memories_str

    async def feedback(self, obs):
        last_action = None
        last_plan = self.current_plan
        # todo: allow many actions in a timestamp.
        for m in self.memory.memories[::-1]:
            if isinstance(m, Action):
                last_action = m
                break
        assert last_action is not None
        assert last_plan is not None
        with LogApiCall(self):
            resp = await async_chat(
                [
                    {"role": "system", "content": self.feedback_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "persona": self.persona,
                                "last_action": last_action.raw_action,
                                "last_plan": last_plan.content,
                                "observation": obs,
                            }
                        ),
                    },
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
                llm_config=self.llm_config,
            )
        resp = json.loads(resp)
        logger.info("feedback: %s", resp)
        for thought in resp["thoughts"]:
            await self.memory.add_memory_piece(Thought(thought, self.memory))

    async def plan(self):
        logger.info("planning ...")
        with LogApiCall(self):
            memories = await self.memory.retrieve(
                self.intent,
                include_recent_observation=True,
                include_recent_action=True,
                include_recent_plan=True,
                include_recent_thought=True,
                trigger_update=False,
                kind_weight={"action": 10, "plan": 10, "thought": 10, "reflection": 10},
            )
            memories = self.format_memories(memories)
            new_plan = ""
            rationale = ""
            while True:
                resp = await async_chat(
                    [
                        {
                            "role": "system",
                            "content": self.planning_prompt,
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "persona": self.persona,
                                    "intent": self.intent,
                                    "memories": memories,
                                    "current_timestamp": self.memory.timestamp,
                                    "old_plan": "N/A"
                                    if self.current_plan is None
                                    else self.current_plan.content,
                                }
                            ),
                        },
                    ],
                    json_mode=True,
                    provider=self.llm_config.provider,
                    model_name=self.llm_config.model,
                    enable_thinking=self.llm_config.enable_thinking,
                    llm_config=self.llm_config,
                )
                # # print(resp)
                resp = resp
                resp = json.loads(resp)
                if "plan" in resp and "rationale" in resp and "next_step" in resp:
                    # logger.info(
                    #     "retrieved memory for planning: %s", "\n".join(memories)
                    # )
                    new_plan = resp["plan"]
                    rationale = resp["rationale"] if "rationale" in resp else "N/A"
                    next_step = resp["next_step"]
                    # make sure they are str
                    if isinstance(new_plan, str) and isinstance(rationale, str):
                        break
                logger.info("invalid response, rethinking... ")
                logger.info("response: %s", resp)
        logger.info("plan: %s", new_plan)
        logger.info("rationale: %s", rationale)
        logger.info("next_step: %s", next_step)
        self.current_plan = Plan(new_plan, self.memory, next_step)
        await self.memory.add_memory_piece(Thought(rationale, self.memory))
        await self.memory.add_memory_piece(self.current_plan)

    async def act(self, env):
        with LogApiCall(self):
            memories = await self.memory.retrieve(
                self.current_plan.next_step,
                trigger_update=False,
                kind_weight={"observation": 0, "action": 10, "thought": 10},
            )
            memories = self.format_memories(memories)
            assert self.current_plan is not None
            clickables = [e for e in env["clickable_elements"] if e is not None]
            inputs = [e for e in env["input_elements"] if e is not None]
            selects = [e for e in env["select_elements"] if e is not None]
            action = await async_chat(
                [
                    {"role": "system", "content": self.action_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "valid_targets": {
                                    "inputs": inputs,
                                    "clickable": clickables,
                                    "selects": selects,
                                },
                                "persona": self.persona,
                                "intent": self.intent,
                                "plan": self.current_plan.content,
                                "next_step": self.current_plan.next_step,
                                "environment": env["html"],
                                "recent_memories": memories,
                            }
                        ),
                    },
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
                llm_config=self.llm_config,
            )
        parsed = json.loads(action)
        logger.info("actions: %s", parsed)
        action = (
            parsed["actions"][0]
            if isinstance(parsed, dict) and "actions" in parsed
            else parsed
        )
        await self.memory.add_memory_piece(
            Action(action["description"], self.memory, json.dumps(action))
        )

        return action


class PersonaAgent:
    """Zero-shot persona-prompted agent.

    A single LLM call per step decides the next concrete action given only
    the persona, the intent, the current page environment, and a short list
    of previous action descriptions. There is no plan, no feedback loop, no
    reference trajectory, and no memory beyond `previous_steps`.
    """

    deny_list = ["Crime", "crime", "Security", "security"]
    _prompt_dir = ["persona_prompts", "ux_prompts"]

    def __init__(self, persona: str, intent: str, llm_config=None) -> None:
        self.action_prompt = load_prompt("action", self._prompt_dir)
        self.persona = persona
        self.intent = intent
        self.llm_config = llm_config
        # Lightweight in-memory bookkeeping. We don't reuse the full Memory
        # graph because zero-shot persona prompting has no observations,
        # plans, or thoughts to retrieve over.
        self.previous_steps: list[str] = []
        self.api_call_count = 0
        self.observation: Optional[str] = None
        # We still construct a Memory so `policy.get_formatted_memories()`
        # has somewhere to read from — actions get pushed into it for trace
        # parity with other agents.
        self.memory = Memory(self)

    async def perceive(self, environment) -> None:
        """Stash the latest environment snapshot. No LLM call: the page
        content is fed directly into the action prompt."""
        if isinstance(environment, str):
            obs = environment
        else:
            obs = json.dumps(environment)
        for denied_word in self.deny_list:
            obs = obs.replace(denied_word, "***")
        self.observation = obs

    async def act(self, env) -> dict:
        with LogApiCall(self):
            clickables = [e for e in env["clickable_elements"] if e is not None]
            inputs = [e for e in env["input_elements"] if e is not None]
            selects = [e for e in env["select_elements"] if e is not None]
            user_content = {
                "persona": self.persona,
                "intent": self.intent,
                "previous_steps": self.previous_steps,
                "environment": {
                    "html": env["html"],
                    "valid_targets": {
                        "inputs": inputs,
                        "clickable": clickables,
                        "selects": selects,
                    },
                },
            }
            resp = await async_chat(
                [
                    {"role": "system", "content": self.action_prompt},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
                llm_config=self.llm_config,
            )
        parsed = json.loads(resp)
        logger.info("persona actions: %s", parsed)
        action = (
            parsed["actions"][0]
            if isinstance(parsed, dict) and "actions" in parsed
            else parsed
        )
        # Track for repetition avoidance and for the memory trace file.
        self.previous_steps.append(action.get("description", json.dumps(action)))
        await self.memory.add_memory_piece(
            Action(action["description"], self.memory, json.dumps(action))
        )
        return action

    # Reuse UXAgent's formatter so trace files keep the same shape across
    # agent variants. (UXAgent.format_memories is already a staticmethod.)
    format_memories = staticmethod(UXAgent.format_memories)


class TrajAgent(UXAgent):
    _prompt_dir = "traj_prompts"

    def __init__(
        self,
        traj: list,
        intent: str,
        llm_config=None,
        persona: str | None = None,
    ) -> None:
        # Initialize UXAgent base with the real persona (may be None when
        # traj_cond runs without the persona sub-module). We don't pass `traj`
        # in as the persona field — that was an earlier bug.
        super().__init__(persona or "", intent, llm_config)
        self.traj = traj
        self.planning_prompt = load_prompt("planning", self._prompt_dir)
        self.feedback_prompt = load_prompt("feedback", self._prompt_dir)
        self.verifier_prompt = load_prompt("verifier", self._prompt_dir)
        self.rethink_prompt = load_prompt("action_rethink", self._prompt_dir)
        self.action_multi_prompt = load_prompt("action_multi", self._prompt_dir)
        self.action_selector_prompt = load_prompt("action_selector", self._prompt_dir)
        self.post_verifier_prompt = load_prompt("verifier_post", self._prompt_dir)

    def _maybe_inject_persona(self, user_content: dict) -> None:
        if self.persona:
            user_content["persona"] = self.persona

    def _aggregate_verifier_result(self, result: dict) -> float:
        criteria = ["executable", "trajectory_consistent", "realistic"]
        total = sum(
            result[c]["score"] * result[c]["confidence"]
            for c in criteria
            if c in result
        )
        return total / len(criteria)

    def passes_verification(self, result: dict) -> bool:
        aggregate_score = self._aggregate_verifier_result(result)
        # breakpoint()
        return aggregate_score >= VERIFIER_PASS_THRESHOLD

    async def commit_action(self, action: dict) -> None:
        await self.memory.add_memory_piece(
            Action(action["description"], self.memory, json.dumps(action))
        )

    async def plan(self, session_sugg: str) -> None:
        logger.info("session agent planning...")
        with LogApiCall(self):
            user_content: dict = {
                "intent": self.intent,
                "sample_trajectory": session_sugg,
                "current_observation": self.observation,
            }
            self._maybe_inject_persona(user_content)
            if self.memory.timestamp > 0:
                last_plan = next(
                    (m for m in reversed(self.memory.memories) if m.kind == "plan"),
                    None,
                )
                last_action = next(
                    (m for m in reversed(self.memory.memories) if m.kind == "action"),
                    None,
                )
                last_thought = next(
                    (m for m in reversed(self.memory.memories) if m.kind == "thought"),
                    None,
                )
                if last_plan:
                    user_content["previous_plan"] = last_plan.content
                if last_action:
                    user_content["previous_action"] = last_action.raw_action
                if last_thought:
                    user_content["previous_feedback"] = last_thought.content

            resp = await async_chat(
                [
                    {"role": "system", "content": self.planning_prompt},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
                llm_config=self.llm_config,
            )
        result = json.loads(resp)
        logger.info("plan: %s", result)
        self.current_plan = Plan(result["plan"], self.memory, result["next_step"])
        await self.memory.add_memory_piece(self.current_plan)
        await self.memory.add_memory_piece(Thought(result["thought"], self.memory))

    async def feedback(self, session_sugg: str, observation: str) -> None:
        last_action = next(
            (m for m in reversed(self.memory.memories) if m.kind == "action"), None
        )
        if last_action is None:
            logger.warning("feedback called but no prior action in memory; skipping")
            return
        logger.info("session agent feedback...")
        with LogApiCall(self):
            resp = await async_chat(
                [
                    {"role": "system", "content": self.feedback_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "sample_trajectory": session_sugg,
                                "action": last_action.raw_action,
                                "new_observation": observation,
                            }
                        ),
                    },
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
                llm_config=self.llm_config,
            )
        result = json.loads(resp)
        logger.info("feedback: %s", result)
        await self.memory.add_memory_piece(Thought(result["thoughts"], self.memory))

    async def act(self, env, session_sugg):
        with LogApiCall(self):
            action_memories = [m for m in self.memory.memories if m.kind == "action"]
            prev_actions = "\n".join(
                self.format_memories(action_memories, sort_by_kind=False)
            )

            clickables = [e for e in env["clickable_elements"] if e is not None]
            inputs = [e for e in env["input_elements"] if e is not None]
            selects = [e for e in env["select_elements"] if e is not None]
            user_content = {
                "valid_targets": {
                    "inputs": inputs,
                    "clickable": clickables,
                    "selects": selects,
                },
                "intent": self.intent,
                "sample_trajectory": session_sugg,
                "previous_steps": prev_actions,
                "environment": env["html"],
            }
            self._maybe_inject_persona(user_content)
            if self.current_plan is not None:
                user_content["plan"] = self.current_plan.content
                user_content["next_step"] = self.current_plan.next_step

            action = await async_chat(
                [
                    {"role": "system", "content": self.action_prompt},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
                llm_config=self.llm_config,
            )
        parsed = json.loads(action)
        logger.info("actions: %s", parsed)
        return (
            parsed["actions"][0]
            if isinstance(parsed, dict) and "actions" in parsed
            else parsed
        )

    async def rethink(self, env, session_sugg, failed_action, verifier_result) -> dict:
        with LogApiCall(self):
            action_memories = [m for m in self.memory.memories if m.kind == "action"]
            prev_actions = "\n".join(
                self.format_memories(action_memories, sort_by_kind=False)
            )
            clickables = [e for e in env["clickable_elements"] if e is not None]
            inputs = [e for e in env["input_elements"] if e is not None]
            selects = [e for e in env["select_elements"] if e is not None]
            user_content = {
                "valid_targets": {
                    "inputs": inputs,
                    "clickable": clickables,
                    "selects": selects,
                },
                "intent": self.intent,
                "sample_trajectory": session_sugg,
                "previous_steps": prev_actions,
                "environment": env["html"],
                "failed_action": failed_action,
                "verifier_feedback": verifier_result,
            }
            self._maybe_inject_persona(user_content)
            if self.current_plan is not None:
                user_content["plan"] = self.current_plan.content
                user_content["next_step"] = self.current_plan.next_step
            action = await async_chat(
                [
                    {"role": "system", "content": self.rethink_prompt},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
                llm_config=self.llm_config,
            )

        parsed = json.loads(action)
        logger.info("rethink action: %s", parsed)
        return (
            parsed["actions"][0]
            if isinstance(parsed, dict) and "actions" in parsed
            else parsed
        )

    async def verify(self, observation, session_sugg, proposed_action) -> dict:
        previous_actions = [
            json.loads(m.raw_action) for m in self.memory.memories if m.kind == "action"
        ]
        action_sequence = previous_actions + [proposed_action]
        # Use the dedicated verifier model when configured, else fall back to the agent's.
        vcfg = getattr(self.llm_config, "verifier_llm", None) or self.llm_config
        with LogApiCall(self):
            resp = await async_chat(
                [
                    {"role": "system", "content": self.verifier_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "intent": self.intent,
                                "sample_trajectory": session_sugg,
                                "current_observation": self.observation,
                                "action_sequence": action_sequence,
                            }
                        ),
                    },
                ],
                json_mode=True,
                provider=vcfg.provider,
                model_name=vcfg.model,
                enable_thinking=vcfg.enable_thinking,
                llm_config=vcfg,
            )
        result = json.loads(resp)
        logger.info("verifier result: %s", result)
        return result

    async def act_multi(self, env, session_sugg) -> list:
        with LogApiCall(self):
            action_memories = [m for m in self.memory.memories if m.kind == "action"]
            prev_actions = "\n".join(
                self.format_memories(action_memories, sort_by_kind=False)
            )
            clickables = [e for e in env["clickable_elements"] if e is not None]
            inputs = [e for e in env["input_elements"] if e is not None]
            selects = [e for e in env["select_elements"] if e is not None]

            user_content = {
                "valid_targets": {
                    "inputs": inputs,
                    "clickable": clickables,
                    "selects": selects,
                },
                "intent": self.intent,
                "sample_trajectory": session_sugg,
                "previous_steps": prev_actions,
                "environment": env["html"],
            }
            if self.current_plan is not None:
                user_content["plan"] = self.current_plan.content
                user_content["next_step"] = self.current_plan.next_step

            resp = await async_chat(
                [
                    {"role": "system", "content": self.action_multi_prompt},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
            )
        result = json.loads(resp)
        candidates = result.get("actions", [])
        logger.info("act_multi candidates (%d): %s", len(candidates), candidates)
        return candidates[:ACT_MULTI_COUNT]

    async def select_action(self, env, session_sugg, candidates: list) -> dict:
        with LogApiCall(self):
            action_memories = [m for m in self.memory.memories if m.kind == "action"]
            prev_actions = "\n".join(
                self.format_memories(action_memories, sort_by_kind=False)
            )
            clickables = [e for e in env["clickable_elements"] if e is not None]
            inputs = [e for e in env["input_elements"] if e is not None]
            selects = [e for e in env["select_elements"] if e is not None]
            user_content = {
                "valid_targets": {
                    "inputs": inputs,
                    "clickable": clickables,
                    "selects": selects,
                },
                "intent": self.intent,
                "sample_trajectory": session_sugg,
                "previous_steps": prev_actions,
                "environment": env["html"],
                "candidate_actions": candidates,
            }
            if self.current_plan is not None:
                user_content["plan"] = self.current_plan.content
                user_content["next_step"] = self.current_plan.next_step

            resp = await async_chat(
                [
                    {"role": "system", "content": self.action_selector_prompt},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                json_mode=True,
                provider=self.llm_config.provider,
                model_name=self.llm_config.model,
                enable_thinking=self.llm_config.enable_thinking,
            )
        result = json.loads(resp)
        selected = result["selected_action"]
        logger.info("select_action chose: %s", selected)
        return selected

    async def post_verify(self, session_sugg, action_trace: list) -> dict:
        # Use the dedicated verifier model when configured, else fall back to the agent's.
        vcfg = getattr(self.llm_config, "verifier_llm", None) or self.llm_config
        with LogApiCall(self):
            resp = await async_chat(
                [
                    {"role": "system", "content": self.post_verifier_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "intent": self.intent,
                                "sample_trajectory": session_sugg,
                                "action_trace": action_trace,
                            }
                        ),
                    },
                ],
                json_mode=True,
                provider=vcfg.provider,
                model_name=vcfg.model,
                enable_thinking=vcfg.enable_thinking,
                llm_config=vcfg,
            )

        result = json.loads(resp)
        logger.info("post_verify result: %s", result)
        return result
