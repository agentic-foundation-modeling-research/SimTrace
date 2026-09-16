import asyncio
import copy
import json
import logging
import os
import pathlib
import shutil
import traceback
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from ..agent import context
from ..executor.env import WebAgentEnv  # Playwright env
from .model import (  # noqa
    TrajAgentPolicy,
    PersonaPolicy,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_STEPS = 30

# start_urls already bootstrapped (login + warmed profile) in THIS process.
# Lets a sequence of experiments that share a start_url reuse the warmed
# session/cache instead of re-running the login bootstrap each time.
_BOOTSTRAPPED_START_URLS: set[str] = set()

_RELATIVE_PATH_FIELDS = [
    "screenshot_path",
    "dom_snapshot_raw",
    "dom_snapshot_simplified",
    "axtree_snapshot",
]


async def _run_for_intent_and_session(
    cfg: DictConfig,
    context_info: Dict,
    start_url: str,
    wait_for_login: bool = True,
    env_setup_hook: Callable = None,
    env_wait_hook: Callable = None,
    user_data_dir_override: Optional[str] = None,
    llm_config=None,
    modules=None,
):
    traj_str = context_info["traj"]
    intent = context_info["intent"]
    session_id = context_info.get("session_id", "")
    use_persona = bool(modules and getattr(modules, "persona", False))
    persona = (
        context_info.get("persona")
        if use_persona
        else None
    )
    log.info(
        f"\n=== traj (first 200 chars) ===\n{traj_str[:200]}...\n=== intent ===\n{intent}"
    )
    trajectory_list = traj_str.split(",")
    run_uid = uuid.uuid4().hex[:8]

    task_to_use = {
        "sites": ["shopping"],
        "task_id": 1,
        "require_login": False,
        "start_url": start_url,
        "intent": intent or "Interactive testing session",
    }

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:4]}"
    base_dir = pathlib.Path(cfg.simulation_config.output_dir).resolve()
    trace_dir = pathlib.Path(os.path.join(base_dir, "runs", run_name))
    # trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "simp_html").mkdir(parents=True, exist_ok=True)
    (trace_dir / "raw_html").mkdir(parents=True, exist_ok=True)
    (trace_dir / "api_trace").mkdir(parents=True, exist_ok=True)
    (trace_dir / "screenshot").mkdir(parents=True, exist_ok=True)
    (trace_dir / "observation_trace").mkdir(parents=True, exist_ok=True)
    (trace_dir / "axtree").mkdir(parents=True, exist_ok=True)
    # save persona and intent
    (trace_dir / "basic_info.json").write_text(json.dumps(context_info))
    context.run_path.set(trace_dir)
    steps_taken = 0

    async def before_action_hook(screenshot_suffix: str = ""):
        if cfg.environment.recording.enabled:
            return
        try:
            # input()
            # save screenshot
            await env.page.screenshot(
                path=trace_dir
                / "screenshot"
                / f"screenshot_{steps_taken}_full_page{screenshot_suffix}.png",
                full_page=True,
                animations="disabled",
                timeout=0,
            )
            await env.page.screenshot(
                path=trace_dir
                / "screenshot"
                / f"screenshot_{steps_taken}{screenshot_suffix}.png",
                animations="disabled",
                timeout=0,
            )
            # scroll position and axtree are highlight-independent; skip on the clean pass
            if screenshot_suffix:
                return
            # get scroll top position
            scroll_top = await env.page.evaluate("window.scrollY")
            with open(
                trace_dir / "screenshot" / f"scroll_top_{steps_taken}.txt", "w"
            ) as f:
                f.write(
                    str(
                        scroll_top
                        * cfg.environment.browser.context_options.device_scale_factor
                    )
                )
            axtree = await env.page.accessibility.snapshot()
            with open(trace_dir / "axtree" / f"axtree_{steps_taken}.json", "w") as f:
                json.dump(axtree, f)
        except Exception as _hook_exc:
            log.error(
                f"[before_action_hook] failed at step {steps_taken}: {_hook_exc!r}",
                exc_info=True,
            )

    if user_data_dir_override is not None:
        env_cfg = copy.deepcopy(cfg.environment)
        env_cfg.browser.user_data_dir = user_data_dir_override
    else:
        env_cfg = cfg.environment

    env = WebAgentEnv(
        env_cfg, before_action_hook=before_action_hook, wait_hook=env_wait_hook
    )

    log.info(f"[{run_uid}] env created")
    session_records = []
    try:
        # based on the different baselines, we need to use the different policies
        effective_llm = llm_config if llm_config is not None else cfg.llm
        effective_modules = (
            modules
            if modules is not None
            else getattr(cfg.simulation_config, "traj_cond", None)
        )
        policy = TrajAgentPolicy(
            trajectory_list,
            intent,
            llm_config=effective_llm,
            modules=effective_modules,
            persona=persona,
        )
        env.memory = policy.agent.memory
        print(
            "setting up env with headless = "
            + str(cfg.environment.browser.launch_options.headless)
        )
        await env.setup(
            task_to_use, headless=cfg.environment.browser.launch_options.headless
        )

        if wait_for_login:
            env.debug_pause()

        # Execute any custom setup actions if specified
        if env_setup_hook:
            await env_setup_hook(env)
        obs = await env.observation()

        log.info("Initial observation ready")

        action_trace = []
        sample_traj_len = len(trajectory_list)
        max_steps = min(int(sample_traj_len * 1.5), MAX_STEPS)
        while steps_taken < max_steps:
            with open(trace_dir / "observation_trace.jsonl", "a") as f:
                json.dump(obs, f)
            if obs.get("tabs"):
                current_url = obs["tabs"][0].get("url")
                print("Current url:", current_url)
            with open(
                trace_dir / "simp_html" / f"simp_html_{steps_taken}.html", "w"
            ) as f:
                f.write(obs["html"])
            with open(
                trace_dir / "raw_html" / f"raw_html_{steps_taken}.html", "w"
            ) as f:
                f.write(await env.page.content())

            # Capture URL and api_call_count before forward() so we know which
            # api_trace files belong to this step and which page the screenshot shows.
            url_at_step = obs["tabs"][0].get("url", "") if obs.get("tabs") else ""
            api_count_before = policy.agent.api_call_count

            # Use our policy to determine the action for this step from the environment
            action = await policy.forward(env, session_sugg=trajectory_list)
            # breakpoint()

            api_count_after = policy.agent.api_call_count
            api_trace_files = [
                f"api_trace/api_trace_{i}.json"
                for i in range(api_count_before + 1, api_count_after + 1)
            ]

            url_map = obs.get("url_map", {})
            try:
                action_dict = json.loads(action)
                target = action_dict.get("target", "")
                if target and target in url_map:
                    action_dict["url"] = url_map[target]
                    action = json.dumps(action_dict)
            except (json.JSONDecodeError, AttributeError):
                pass

            try:
                clicked_url = json.loads(action).get("url", "")
            except (json.JSONDecodeError, AttributeError):
                clicked_url = ""

            action_trace.append(action)
            with open(trace_dir / "action_trace.json", "w") as f:
                json.dump(action_trace, f, indent=2)
            with open(
                trace_dir
                / "observation_trace"
                / f"observation_trace_{steps_taken}.txt",
                "w",
            ) as f:
                f.write(str(policy.agent.observation))
            # save memory trace
            # with open(trace_dir / "memory_trace.json", "w") as f:
            #     json.dump(policy.agent.memory.memories, f)
            print(f"Taking action {action}")
            print(f"Action: {steps_taken + 1} out of {max_steps}")

            step_timestamp = datetime.now().isoformat()
            obs = await env.step(action)
            steps_taken += 1

            session_records.append(
                {
                    "session_id": session_id,
                    "timestamp": step_timestamp,
                    "synthetic_action": action,
                    "clicked_url": clicked_url,
                    "url": url_at_step,
                    "llm_call": api_trace_files,
                    "screenshot_path": f"screenshot/screenshot_{steps_taken - 1}.png",
                    "dom_snapshot_raw": f"raw_html/raw_html_{steps_taken - 1}.html",
                    "dom_snapshot_simplified": f"simp_html/simp_html_{steps_taken - 1}.html",
                    "axtree_snapshot": f"axtree/axtree_{steps_taken - 1}.json",
                }
            )

            if obs.get("terminated"):
                break
            
            # we only care about the first checkout action, in the real data, everything after first checkout also dropped.
            if "/checkout" in clicked_url or "/checkouts/" in clicked_url:
                break

        log.info(
            f"Finished persona run: terminated={obs.get('terminated')}, "
            f"score={obs.get('score')}, steps={steps_taken}"
        )

        # breakpoint()
        # ---- post-session verifier (verifier: "post") ----
        verifier_mode = str(getattr(effective_modules, "verifier", "no"))
        if verifier_mode == "post" and action_trace:
            action_trace_objects = [json.loads(a) for a in action_trace]
            post_result = await policy.agent.post_verify(
                trajectory_list, action_trace_objects
            )
            with open(trace_dir / "post_verify_result.json", "w") as f:
                json.dump(post_result, f, indent=2)
            log.info("Post-session verification result: %s", post_result)

        # ---- save final memory trace ----
        final_memories_str = policy.get_formatted_memories()

        trace_file = trace_dir / f"{run_name}.txt"
        trace_file.write_text(final_memories_str, encoding="utf-8")

        log.info(f"Saved memory trace to {trace_file}")

        with open(trace_dir / "session_data.json", "w") as f:
            json.dump(session_records, f, indent=2)
        log.info(
            f"Saved {len(session_records)} step records to {trace_dir}/session_data.json"
        )

    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            (trace_dir / "error.txt").write_text(err)
            if session_records:
                with open(trace_dir / "session_data.json", "w") as f:
                    json.dump(session_records, f, indent=2)
        except Exception:
            pass
    finally:
        try:
            log.info(f"[{run_uid}] closing env...")
            await asyncio.wait_for(asyncio.shield(env.close()), timeout=10)
            log.info(f"[{run_uid}] env.close() completed")
        except Exception as e:
            log.exception(f"[{run_uid}] env.close() raised: {e!r}")
    return trace_dir, session_records


async def _run_for_persona_zero_shot(
    cfg: DictConfig,
    context_info: Dict,
    start_url: str,
    wait_for_login: bool = True,
    env_setup_hook: Callable = None,
    env_wait_hook: Callable = None,
    user_data_dir_override: Optional[str] = None,
    llm_config=None,
):
    """Runner for agent_mode=persona — zero-shot persona prompting.

    Each step makes a single LLM call given (persona, intent, environment,
    previous_steps) and returns one action. The loop terminates when the
    agent emits a `terminate` action or MAX_STEPS is reached.
    """
    persona = context_info.get("persona")
    intent = context_info["intent"]
    session_id = context_info.get("session_id", "")
    log.info(
        f"\n=== persona (first 200 chars) ===\n{persona[:200]}...\n=== intent ===\n{intent}"
    )
    run_uid = uuid.uuid4().hex[:8]

    task_to_use = {
        "sites": ["shopping"],
        "task_id": 1,
        "require_login": False,
        "start_url": start_url,
        "intent": intent or "Interactive testing session",
    }

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:4]}"
    base_dir = pathlib.Path(cfg.simulation_config.output_dir).resolve()
    trace_dir = pathlib.Path(os.path.join(base_dir, "runs", run_name))
    (trace_dir / "simp_html").mkdir(parents=True, exist_ok=True)
    (trace_dir / "raw_html").mkdir(parents=True, exist_ok=True)
    (trace_dir / "api_trace").mkdir(parents=True, exist_ok=True)
    (trace_dir / "screenshot").mkdir(parents=True, exist_ok=True)
    (trace_dir / "observation_trace").mkdir(parents=True, exist_ok=True)
    (trace_dir / "axtree").mkdir(parents=True, exist_ok=True)
    (trace_dir / "basic_info.json").write_text(json.dumps(context_info))
    context.run_path.set(trace_dir)
    steps_taken = 0

    async def before_action_hook(screenshot_suffix: str = ""):
        if cfg.environment.recording.enabled:
            return
        try:
            await env.page.screenshot(
                path=trace_dir
                / "screenshot"
                / f"screenshot_{steps_taken}_full_page{screenshot_suffix}.png",
                full_page=True,
                animations="disabled",
                timeout=0,
            )
            await env.page.screenshot(
                path=trace_dir
                / "screenshot"
                / f"screenshot_{steps_taken}{screenshot_suffix}.png",
                animations="disabled",
                timeout=0,
            )
            # scroll position and axtree are highlight-independent; skip on the clean pass
            if screenshot_suffix:
                return
            scroll_top = await env.page.evaluate("window.scrollY")
            with open(
                trace_dir / "screenshot" / f"scroll_top_{steps_taken}.txt", "w"
            ) as f:
                f.write(
                    str(
                        scroll_top
                        * cfg.environment.browser.context_options.device_scale_factor
                    )
                )
            axtree = await env.page.accessibility.snapshot()
            with open(trace_dir / "axtree" / f"axtree_{steps_taken}.json", "w") as f:
                json.dump(axtree, f)
        except Exception as _hook_exc:
            log.error(
                f"[before_action_hook] failed at step {steps_taken}: {_hook_exc!r}",
                exc_info=True,
            )

    if user_data_dir_override is not None:
        env_cfg = copy.deepcopy(cfg.environment)
        env_cfg.browser.user_data_dir = user_data_dir_override
    else:
        env_cfg = cfg.environment

    env = WebAgentEnv(
        env_cfg, before_action_hook=before_action_hook, wait_hook=env_wait_hook
    )

    log.info(f"[{run_uid}] env created")
    session_records = []
    try:
        effective_llm = llm_config if llm_config is not None else cfg.llm
        policy = PersonaPolicy(persona, intent, llm_config=effective_llm)
        env.memory = policy.agent.memory
        print(
            "setting up env with headless = "
            + str(cfg.environment.browser.launch_options.headless)
        )
        await env.setup(
            task_to_use, headless=cfg.environment.browser.launch_options.headless
        )

        if wait_for_login:
            env.debug_pause()

        if env_setup_hook:
            await env_setup_hook(env)
        obs = await env.observation()

        log.info("Initial observation ready")

        action_trace = []
        max_steps = MAX_STEPS
        while steps_taken < max_steps:
            with open(trace_dir / "observation_trace.jsonl", "a") as f:
                json.dump(obs, f)
            if obs.get("tabs"):
                current_url = obs["tabs"][0].get("url")
                print("Current url:", current_url)
            with open(
                trace_dir / "simp_html" / f"simp_html_{steps_taken}.html", "w"
            ) as f:
                f.write(obs["html"])
            with open(
                trace_dir / "raw_html" / f"raw_html_{steps_taken}.html", "w"
            ) as f:
                f.write(await env.page.content())

            url_at_step = obs["tabs"][0].get("url", "") if obs.get("tabs") else ""
            api_count_before = policy.agent.api_call_count

            action = await policy.forward(env)

            api_count_after = policy.agent.api_call_count
            api_trace_files = [
                str(trace_dir / "api_trace" / f"api_trace_{i}.json")
                for i in range(api_count_before + 1, api_count_after + 1)
            ]

            url_map = obs.get("url_map", {})
            try:
                action_dict = json.loads(action)
                target = action_dict.get("target", "")
                if target and target in url_map:
                    action_dict["url"] = url_map[target]
                    action = json.dumps(action_dict)
            except (json.JSONDecodeError, AttributeError):
                pass

            try:
                clicked_url = json.loads(action).get("url", "")
            except (json.JSONDecodeError, AttributeError):
                clicked_url = ""

            action_trace.append(action)
            with open(trace_dir / "action_trace.json", "w") as f:
                json.dump(action_trace, f, indent=2)
            with open(
                trace_dir
                / "observation_trace"
                / f"observation_trace_{steps_taken}.txt",
                "w",
            ) as f:
                f.write(str(policy.agent.observation))
            print(f"Taking action {action}")
            print(f"Action: {steps_taken + 1} out of {max_steps}")

            step_timestamp = datetime.now().isoformat()
            obs = await env.step(action)
            steps_taken += 1

            session_records.append(
                {
                    "session_id": session_id,
                    "timestamp": step_timestamp,
                    "synthetic_action": action,
                    "clicked_url": clicked_url,
                    "url": url_at_step,
                    "llm_call": api_trace_files,
                    "screenshot_path": str(
                        trace_dir / "screenshot" / f"screenshot_{steps_taken - 1}.png"
                    ),
                    "dom_snapshot_raw": str(
                        trace_dir / "raw_html" / f"raw_html_{steps_taken - 1}.html"
                    ),
                    "dom_snapshot_simplified": str(
                        trace_dir / "simp_html" / f"simp_html_{steps_taken - 1}.html"
                    ),
                    "axtree_snapshot": str(
                        trace_dir / "axtree" / f"axtree_{steps_taken - 1}.json"
                    ),
                }
            )

            if obs.get("terminated"):
                break

        log.info(
            f"Finished persona zero-shot run: terminated={obs.get('terminated')}, "
            f"score={obs.get('score')}, steps={steps_taken}"
        )

        final_memories_str = policy.get_formatted_memories()
        trace_file = trace_dir / f"{run_name}.txt"
        trace_file.write_text(final_memories_str, encoding="utf-8")
        log.info(f"Saved memory trace to {trace_file}")

        with open(trace_dir / "session_data.json", "w") as f:
            json.dump(session_records, f, indent=2)
        log.info(
            f"Saved {len(session_records)} step records to {trace_dir}/session_data.json"
        )

    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            (trace_dir / "error.txt").write_text(err)
            if session_records:
                with open(trace_dir / "session_data.json", "w") as f:
                    json.dump(session_records, f, indent=2)
        except Exception:
            pass
    finally:
        try:
            log.info(f"[{run_uid}] closing env...")
            await asyncio.wait_for(asyncio.shield(env.close()), timeout=10)
            log.info(f"[{run_uid}] env.close() completed")
        except Exception as e:
            log.exception(f"[{run_uid}] env.close() raised: {e!r}")
    return trace_dir, session_records


def _load_cfg(config_name: str = "base"):
    here = pathlib.Path(__file__).resolve().parent
    conf_dir = here.parents[2] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name=config_name)
    return cfg


def combine_session_data(output_dir: str) -> pathlib.Path | None:
    """Scan all completed runs under <output_dir>/runs/ and write a combined
    session_data_{run_ts}.json at the output_dir root. Returns the output path,
    or None if no records were found."""
    base_dir = pathlib.Path(output_dir).resolve()
    runs_dir = base_dir / "runs"
    combined_records: list = []
    if runs_dir.exists():
        for folder in sorted(runs_dir.iterdir()):
            if not folder.is_dir():
                continue
            session_data_path = folder / "session_data.json"
            error_path = folder / "error.txt"
            if session_data_path.exists() and not error_path.exists():
                try:
                    with open(session_data_path) as f:
                        records = json.load(f)
                    run_id = folder.name
                    for record in records:
                        for field in _RELATIVE_PATH_FIELDS:
                            if (
                                record.get(field)
                                and not pathlib.Path(record[field]).is_absolute()
                            ):
                                record[field] = f"runs/{run_id}/{record[field]}"
                        if "llm_call" in record:
                            record["llm_call"] = [
                                f"runs/{run_id}/{p}"
                                if not pathlib.Path(p).is_absolute()
                                else p
                                for p in record["llm_call"]
                            ]
                    combined_records.extend(records)
                except Exception as e:
                    log.warning(f"Failed to load {session_data_path}: {e}")

    if not combined_records:
        log.warning(f"No completed run records found under {runs_dir}")
        return None

    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = base_dir / f"session_data_{run_ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined_records, f, indent=2)
    log.info(f"Saved {len(combined_records)} total step records to {out_path}")
    return out_path


async def experiment_async(
    agents: List[Dict[str, str]],
    start_url: str,
    max_steps: int = 50,
    *,
    headless=False,
    config_name: str = "base",
    config_path: str = ".",
    concurrency: int = 4,
    llm_config=None,
    output_dir: str | None = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    modules=None,
    agent_mode: str | None = None,
) -> tuple[list, pathlib.Path | None]:
    cfg = _load_cfg(config_name=config_name)
    cfg.environment.browser.launch_options.headless = headless
    if output_dir is not None:
        cfg.simulation_config.output_dir = output_dir
    effective_llm = llm_config if llm_config is not None else cfg.llm
    print("llm provider: " + effective_llm.provider)

    # Prefer the explicitly-passed agent_mode (e.g. from a sweep override); fall back to
    # the value composed from conf/base.yaml when not provided.
    agent_mode = agent_mode or cfg.simulation_config.agent_mode

    # Pre-create one fresh user_data_dir copy per conversation so that
    # each session starts clean with no prior browser or chat state.
    original_user_data_dir = OmegaConf.select(cfg, "environment.browser.user_data_dir")
    if original_user_data_dir:
        original_path = pathlib.Path(original_user_data_dir)
        # Sibling marker (not inside the profile, so it is not copied into workers)
        # recording which start_url the master profile was bootstrapped for.
        marker_path = original_path.parent / f"{original_path.name}.bootstrap_url"

        # Reuse the warmed master profile (session cookies + cache) when we already
        # bootstrapped this exact start_url earlier in THIS process and the on-disk
        # master still matches. The in-process guard forces a clean bootstrap on the
        # first experiment of each run, so a stale profile is never silently reused.
        reuse = (
            start_url in _BOOTSTRAPPED_START_URLS
            and original_path.exists()
            and marker_path.exists()
            and marker_path.read_text().strip() == start_url
        )

        if reuse:
            log.info(
                "Reusing bootstrapped browser session/cache for start_url=%s "
                "(skipping login bootstrap).",
                start_url,
            )
        else:
            # clear the existing user_data_dir to ensure a clean slate for the bootstrap login
            if os.path.exists(original_user_data_dir):
                shutil.rmtree(original_user_data_dir)
            # Bootstrap: do a one-time headful login
            login_cfg = copy.deepcopy(cfg.environment)
            login_cfg.browser.user_data_dir = str(original_path)
            login_env = WebAgentEnv(login_cfg)
            if login_cfg.browser.launch_options.headless:
                log.warning("Login environment is running in headless mode. ")
            await login_env.setup(
                {
                    "sites": ["shopping"],
                    "task_id": 1,
                    "require_login": True,
                    "start_url": start_url,
                    "intent": "Login",
                },
                headless=login_cfg.browser.launch_options.headless,
            )
            
            if not login_cfg.browser.launch_options.headless:
                login_env.debug_pause()

            await login_env.close()
            # Record what the master now holds so later experiments can reuse it.
            marker_path.write_text(start_url)
            _BOOTSTRAPPED_START_URLS.add(start_url)
        # Create num_workers fresh copies — recycled across all conversations
        available_dirs: list = []
        for i in range(concurrency):
            worker_path = original_path.parent / f"{original_path.name}_conv_{i}"
            if worker_path.exists():
                shutil.rmtree(str(worker_path))
            if original_path.exists():
                shutil.copytree(str(original_path), str(worker_path))
            available_dirs.append(str(worker_path))
        task_to_dir: dict = {}
    else:
        available_dirs = None
        task_to_dir = None

    # Run conversations with limited concurrency using asyncio.wait
    running_tasks = set()
    results = []

    total = len(agents)
    done = 0
    lock = asyncio.Lock()

    def _collect_result(task):
        result = task.result()
        if isinstance(result, tuple):
            trace_dir_r, _ = result
            results.append(trace_dir_r)
        else:
            results.append(result)

    for entry in agents:
        # Throttle: wait for at least one task to finish before dispatching the next
        if len(running_tasks) >= concurrency:
            finished, running_tasks = await asyncio.wait(
                running_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished:
                # Return the worker dir to the pool so the next dispatch can claim it
                if task_to_dir is not None and task in task_to_dir:
                    returned_dir = task_to_dir.pop(task)
                    # Re-fresh this slot so the next batch of conversations get a clean profile
                    if original_path.exists() and returned_dir:
                        if os.path.exists(returned_dir):
                            shutil.rmtree(returned_dir)
                        shutil.copytree(str(original_path), returned_dir)
                    available_dirs.append(returned_dir)
                try:
                    _collect_result(task)
                except Exception as e:
                    log.exception("A session failed", exc_info=e)
                async with lock:
                    done += 1
                    if on_progress:
                        try:
                            on_progress(done, total)
                        except Exception:
                            pass

        worker_dir = available_dirs.pop(0) if available_dirs else None


        if agent_mode == "traj_cond":  # session_follow
            coro = _run_for_intent_and_session(
                cfg=cfg,
                context_info=entry,
                start_url=start_url,
                wait_for_login=cfg.environment.browser.get("wait_for_login", True),
                user_data_dir_override=worker_dir,
                llm_config=effective_llm,
                modules=modules,
            )
        elif agent_mode == "persona":  # zero-shot persona prompting
            coro = _run_for_persona_zero_shot(
                cfg=cfg,
                context_info=entry,
                start_url=start_url,
                wait_for_login=cfg.environment.browser.get("wait_for_login", True),
                user_data_dir_override=worker_dir,
                llm_config=effective_llm,
            )

        task = asyncio.create_task(coro)
        if task_to_dir is not None and worker_dir is not None:
            task_to_dir[task] = worker_dir
        running_tasks.add(task)

    # Drain any remaining in-flight tasks
    if running_tasks:
        finished, _ = await asyncio.wait(running_tasks)
        for task in finished:
            if task_to_dir is not None and task in task_to_dir:
                available_dirs.append(task_to_dir.pop(task))
            try:
                _collect_result(task)
            except Exception as e:
                log.exception("A session failed", exc_info=e)
            async with lock:
                done += 1
                if on_progress:
                    try:
                        on_progress(done, total)
                    except Exception:
                        pass

    combined_path = combine_session_data(cfg.simulation_config.output_dir)

    return results, combined_path
