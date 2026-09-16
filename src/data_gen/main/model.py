import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Optional


from ..agent import TrajAgent, PersonaAgent
from ..agent.constants import VERIFIER_MAX_ATTEMPTS
from ..executor.env import WebAgentEnv

logger = logging.getLogger(__name__)


class BasePolicy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def forward(self, playwright_env: WebAgentEnv):
        """
        Args:
            playwright_env:
                WebAgentEnv object representing the current playwright environment from which
                observation can be drawn.

        Returns:
            action (`str`):
                Return serializable string of the format '{"action": <action>, ...}'
                Examples:
                    '{"action": "click", "target": "login_button"}'
                    '{"action": "select", "target": "country", "value": "US"}'
        """
        raise NotImplementedError


class PersonaPolicy(BasePolicy):
    """Policy wrapping the zero-shot PersonaAgent.

    One LLM call per step: the agent receives the persona, intent, the
    current page environment, and the list of prior action descriptions,
    and emits a single next action. No memory loop, no plan, no feedback.
    """

    def __init__(self, persona, intent, output=None, llm_config=None):
        logger.info(
            f"Creating PersonaPolicy with persona (first 80 chars): {persona[:80]!r}, "
            f"intent: {intent}"
        )
        self.agent = PersonaAgent(persona, intent, llm_config=llm_config)
        self.persona = persona
        self.intent = intent

    async def forward(self, playwright_env, **kwargs):
        observation = await playwright_env.observation()
        # Cheap perceive: stash the html snapshot for tracing only.
        await self.agent.perceive(observation["html"])
        action = await self.agent.act(observation)
        self.agent.memory.timestamp += 1
        return json.dumps(action)

    def get_formatted_memories(self) -> str:
        if not self.agent.memory.memories:
            return ""
        return "\n".join(self.agent.format_memories(self.agent.memory.memories))

    async def close(self):
        # Nothing to clean up — no slow_loop, no background tasks.
        return


class TrajAgentPolicy(BasePolicy):
    def __init__(
        self,
        traj: List,
        intent,
        output=None,
        llm_config=None,
        modules=None,
        persona: Optional[str] = None,
    ):
        self.agent = TrajAgent(traj, intent, llm_config=llm_config, persona=persona)
        self.traj = traj
        self.intent = intent
        self.persona = persona
        self.modules = modules

    async def forward(self, playwright_env, **kwarg):
        observation = await playwright_env.observation()
        session_sugg = kwarg.get("session_sugg")

        # breakpoint()
        if (
            self.modules is not None
            and self.modules.feedback
            and self.agent.memory.timestamp > 0
        ):
            await self.agent.feedback(session_sugg, observation["html"])

        # breakpoint()

        await self.agent.perceive(observation["html"])

        # breakpoint()
        if self.modules is not None and self.modules.plan:
            await self.agent.plan(session_sugg)

        # breakpoint()

        act_mode = getattr(self.modules, "act", "single") if self.modules else "single"
        verifier_mode = (
            getattr(self.modules, "verifier", "no") if self.modules else "no"
        )

        action = None
        verify_result = None
        for attempt in range(VERIFIER_MAX_ATTEMPTS):
            if attempt == 0 or verify_result is None:
                if act_mode == "multi":
                    candidates = await self.agent.act_multi(observation, session_sugg)
                    # breakpoint()
                    action = await self.agent.select_action(
                        observation, session_sugg, candidates
                    )
                elif act_mode == "single":
                    action = await self.agent.act(observation, session_sugg)
            else:
                action = await self.agent.rethink(
                    observation, session_sugg, action, verify_result
                )
            # breakpoint()
            if verifier_mode != "pre":
                break
            verify_result = await self.agent.verify(observation, session_sugg, action)
            # breakpoint()
            if self.agent.passes_verification(verify_result):
                logger.info("verifier passed on attempt %d", attempt + 1)
                break
            logger.info(
                "verifier failed attempt %d/%d, re-generating action",
                attempt + 1,
                VERIFIER_MAX_ATTEMPTS,
            )

        await self.agent.commit_action(action)
        self.agent.memory.timestamp += 1
        return json.dumps(action)

    def get_formatted_memories(self) -> str:
        """
        Return all memories of the agent as a single formatted string.

        Returns:
            str: Formatted memory trace.
        """
        if not self.agent.memory.memories:
            return ""
        return "\n".join(self.agent.format_memories(self.agent.memory.memories))

    async def close(self):
        if self.slow_loop_task is not None:
            self.slow_loop_task.cancel()
            self.slow_loop_task = None

