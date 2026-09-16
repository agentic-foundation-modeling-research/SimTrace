import asyncio
import base64
import contextlib
import inspect
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

import numpy
from omegaconf import DictConfig, OmegaConf
from playwright.async_api import Playwright, async_playwright
from playwright_stealth.stealth import Stealth


class ElementHighlight:
    """
    Context manager for highlighting elements using overlay instead of modifying target element.
    Playwright-based implementation that creates a visual overlay on top of the target element.
    """

    def __init__(
        self,
        semantic_id: Optional[str] = None,
        sleep: float = 1,
        before_hook: Optional[str] = None,
        after_hook: Optional[str] = None,
        center: bool = True,
    ):
        # Automatically get the WebAgentEnv instance from the call stack
        self.env = self._get_env_from_stack()
        self.page = self.env.page
        self.semantic_id = semantic_id
        self.sleep_duration = sleep
        self.before_hook = before_hook
        self.after_hook = after_hook
        self.center = center
        self.logger = logging.getLogger(__name__)

        # Get headless mode from environment config
        try:
            self.headless = bool(
                getattr(self.env.config.browser.launch_options, "headless", False)
            )
        except Exception:
            self.headless = False

    def _get_env_from_stack(self):
        """
        Extract the WebAgentEnv instance from the call stack.
        Looks for 'self' in the calling frame that is an instance of WebAgentEnv.
        """
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the WebAgentEnv instance
            while frame:
                frame = frame.f_back
                if frame is None:
                    break

                # Check if 'self' exists in the frame's local variables
                if "self" in frame.f_locals:
                    potential_env = frame.f_locals["self"]
                    # Check if it's a WebAgentEnv instance using isinstance
                    if isinstance(potential_env, WebAgentEnv):
                        return potential_env

            raise RuntimeError(
                "Could not find WebAgentEnv instance in call stack. "
                "ElementHighlight must be called from within a WebAgentEnv method."
            )
        finally:
            # Clean up frame reference to avoid memory leaks
            del frame

    async def __aenter__(self):
        try:
            # Only perform highlighting and scrolling if we have a target element
            if self.semantic_id:
                # Get the target element
                selector = f'[parser-semantic-id="{self.semantic_id}"]'
                locator = self.page.locator(selector).first

                # Ensure element exists
                count = await locator.count()
                if count == 0:
                    self.logger.warning(
                        f'Element with parser-semantic-id="{self.semantic_id}" not found, skipping highlight'
                    )
                else:
                    # Scroll element into view with smooth behavior
                    if self.center:
                        await self.page.evaluate(
                            """(sel) => {
                            const el = document.querySelector(sel);
                            if (!el) return;
                            requestAnimationFrame(() => {
                                    el.scrollIntoView({behavior: "smooth", block: "center", inline: "center"});
                            });
                        }""",
                            selector,
                        )
                    else:
                        await locator.scroll_into_view_if_needed(timeout=2000)
                    await asyncio.sleep(1)

                    # Capture a clean screenshot of the same frame before drawing the overlay
                    if self.env.before_action_hook:
                        if inspect.iscoroutinefunction(self.env.before_action_hook):
                            await self.env.before_action_hook(
                                screenshot_suffix="_pre_highlight"
                            )
                        else:
                            self.env.before_action_hook(
                                screenshot_suffix="_pre_highlight"
                            )

                    # Create dual overlay highlight - inner (exact) and outer (with gap)
                    await self.page.evaluate(
                        """(sel) => {
                    const element = document.querySelector(sel);
                    if (!element) return;
                    
                    // Function to calculate cumulative offset (same as Selenium version)
                    var cumulativeOffset = function(element) {
                        var top = 0, left = 0;
                        var rect = element.getBoundingClientRect();
                        do {
                            top += element.offsetTop || 0;
                            left += element.offsetLeft || 0;
                            element = element.offsetParent;
                        } while(element);

                        return {
                            top: top,
                            left: left,
                            width: rect.width,
                            height: rect.height,
                        };
                    };
                    
                    // Get element position and dimensions
                    const rect = cumulativeOffset(element);
                    
                    // Create outer highlight div (larger with gap)
                    const gap = 16; // 8px gap around the element
                    const outerDiv = document.createElement('div');
                    outerDiv.style.position = 'absolute';
                    outerDiv.style.top = (rect.top - gap) + 'px';
                    outerDiv.style.left = (rect.left - gap) + 'px';
                    outerDiv.style.width = (rect.width + gap * 2) + 'px';
                    outerDiv.style.height = (rect.height + gap * 2) + 'px';
                    outerDiv.style.outline = '5px solid rgba(121, 204, 215, 0.6)';
                    outerDiv.style.zIndex = '2147483646';
                    outerDiv.style.pointerEvents = 'none';
                    
                    // Create inner highlight div (exact size)
                    const innerDiv = document.createElement('div');
                    innerDiv.style.position = 'absolute';
                    innerDiv.style.top = rect.top + 'px';
                    innerDiv.style.left = rect.left + 'px';
                    innerDiv.style.width = rect.width + 'px';
                    innerDiv.style.height = rect.height + 'px';
                    innerDiv.style.outline = '3px solid #79ccd7';
                    innerDiv.style.zIndex = '2147483647';
                    innerDiv.style.pointerEvents = 'none';
                    
                    // Add both to DOM
                    document.body.appendChild(outerDiv);
                    document.body.appendChild(innerDiv);
                    
                    // Store references for cleanup
                    document.highlightedElements = [outerDiv, innerDiv];
                        }""",
                        selector,
                    )

                    self.logger.info("Element highlight applied")

            # Always sleep (whether we have a target or not)
            if self.sleep_duration > 0:
                await asyncio.sleep(self.sleep_duration)

            # Run before hook if provided and we have a target element
            if self.before_hook and self.semantic_id:
                selector = f'[parser-semantic-id="{self.semantic_id}"]'
                locator = self.page.locator(selector).first
                try:
                    await locator.evaluate(self.before_hook)
                except Exception as e:
                    self.logger.warning(f"Before hook failed: {e}")

            # Always run environment before action hook
            if self.env.before_action_hook:
                if inspect.iscoroutinefunction(self.env.before_action_hook):
                    await self.env.before_action_hook()
                else:
                    self.env.before_action_hook()

            # Run this after the first snippet
            await self.page.evaluate(
                """() => {
                const refs = document.highlightedElements;
                if (!refs || refs.length < 2) return Promise.resolve();

                const [outerDiv, innerDiv] = refs;

                // Parse target box (the exact element bounds already stored on innerDiv)
                const tTop = parseFloat(innerDiv.style.top || '0');
                const tLeft = parseFloat(innerDiv.style.left || '0');
                const tWidth = parseFloat(innerDiv.style.width || '0');
                const tHeight = parseFloat(innerDiv.style.height || '0');

                // Prepare CSS transition for a smooth 2s shrink
                outerDiv.style.transition = [
                    'top 2000ms ease',
                    'left 2000ms ease',
                    'width 2000ms ease',
                    'height 2000ms ease',
                    'outline-width 2000ms ease',
                    'outline-color 2000ms ease'
                ].join(', ');
                outerDiv.style.willChange = 'top,left,width,height,outline-width,outline-color';

                // Force reflow so transition reliably applies
                void outerDiv.offsetWidth;

                // Animate: shrink outer box to the inner box
                outerDiv.style.top = tTop + 'px';
                outerDiv.style.left = tLeft + 'px';
                outerDiv.style.width = tWidth + 'px';
                outerDiv.style.height = tHeight + 'px';

                // Return a promise that resolves when transition ends
                return new Promise(resolve => {
                    const handler = (e) => {
                        if (e.target === outerDiv) {
                            outerDiv.removeEventListener('transitionend', handler);
                            resolve();
                        }
                    };
                    outerDiv.addEventListener('transitionend', handler);
                });
            }
            """
            )

        except Exception as e:
            self.logger.error(f"Failed to apply element highlight: {e}")
            # Continue without highlighting rather than failing

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        try:
            # Remove highlight overlays only if we have a target and not in headless mode
            if self.semantic_id:
                await self.page.evaluate(
                    """
                    if (document.highlightedElements) {
                        document.highlightedElements.forEach(el => el && el.remove());
                        document.highlightedElements = null;
                    }
                    // Fallback for old single element cleanup
                    if (document.highlightedElement) {
                        document.highlightedElement.remove();
                        document.highlightedElement = null;
                    }
                    """
                )

            # Run after hook if provided and we have a target element
            if self.after_hook and self.semantic_id:
                selector = f'[parser-semantic-id="{self.semantic_id}"]'
                locator = self.page.locator(selector).first
                try:
                    await locator.evaluate(self.after_hook)
                except Exception as e:
                    self.logger.warning(f"After hook failed: {e}")

            # Always run environment after action hook
            if self.env.after_action_hook:
                if inspect.iscoroutinefunction(self.env.after_action_hook):
                    await self.env.after_action_hook()
                else:
                    self.env.after_action_hook()

        except Exception as e:
            self.logger.warning(f"Failed to remove element highlight: {e}")
            # Don't raise - cleanup failures shouldn't break the flow

    def pause(self, duration: float = None) -> float:
        """Generate a pause duration with some randomness (similar to Selenium version)"""
        if duration is None:
            duration = max(0.2 + numpy.random.normal(0, 0.05), 0)
        return duration

    async def sleep(self, duration: float = None):
        """Async sleep with randomness"""
        if self.headless:
            return
        sleep_time = self.pause(duration)
        await asyncio.sleep(sleep_time)


class WebAgentEnv:
    _shared_playwright: ClassVar[Playwright | None] = None
    _shared_playwright_users: ClassVar[int] = 0
    _playwright_lock: ClassVar[asyncio.Lock | None] = None
    before_action_hook: Callable[[], None] = None
    after_action_hook: Callable[[], None] = (None,)
    wait_hook: Callable[[], None] = None

    def __init__(
        self,
        environment_config: DictConfig,
        before_action_hook: Callable[[], None] = None,
        after_action_hook: Callable[[], None] = None,
        wait_hook: Callable[[], None] = None,
    ):
        self.config = environment_config
        self.before_action_hook = before_action_hook
        self.after_action_hook = after_action_hook
        self.wait_hook = wait_hook
        self.context_manager = None
        self.browser = None
        self.context = None
        self.page = None  # current active page
        # note: pages are managed by self.context.pages
        self.uuid = (
            environment_config.uuid
            if hasattr(environment_config, "uuid")
            else str(uuid.uuid4())
        )
        self.logger = logging.getLogger(__name__)
        self.task_config: dict | None = None
        self.model_answer: str | None = None  # Model's final answer/response
        self.trace_file_path: str | None = None  # Path to the current trace file

        # Disable evaluation if recording is enabled
        if getattr(self.config, "recording", {}).get("enabled", False):
            if hasattr(self.config, "evaluation"):
                self.config.evaluation.enabled = False
            self.logger.warning(
                "Recording is enabled - evaluation disabled to avoid interference"
            )

    @classmethod
    def _reset_shared_state(cls) -> None:
        """Drop process-wide Playwright state that is bound to a (now-closed) event
        loop. Called once per fresh asyncio.run loop so the lock/shared instance
        re-bind to the current loop instead of raising 'bound to a different event
        loop'. Best-effort: a stale Playwright from a crashed run can't be stopped
        cleanly here, so just drop the references."""
        cls._playwright_lock = None
        cls._shared_playwright = None
        cls._shared_playwright_users = 0

    @classmethod
    def _ensure_loop_bound_lock(cls) -> asyncio.Lock:
        """Return the shared lock, recreating it if it is missing or bound to a
        different (e.g. closed) event loop. An ``asyncio.Lock`` permanently binds
        to the loop active when first used; reusing one across ``asyncio.run``
        boundaries raises 'bound to a different event loop' on contended acquire."""
        running_loop = asyncio.get_running_loop()
        lock = cls._playwright_lock
        if lock is None:
            cls._playwright_lock = asyncio.Lock()
            return cls._playwright_lock
        try:
            bound_loop = lock._get_loop()  # private, but the only binding accessor
        except Exception:
            bound_loop = None
        if bound_loop is not running_loop:
            cls._playwright_lock = asyncio.Lock()
        return cls._playwright_lock

    @classmethod
    async def _ensure_playwright(cls) -> Playwright:
        """Ensure shared Playwright instance exists and return it"""
        async with cls._ensure_loop_bound_lock():
            if cls._shared_playwright is None:
                cls._shared_playwright = await async_playwright().start()
            cls._shared_playwright_users += 1
            return cls._shared_playwright

    @classmethod
    async def _cleanup_playwright(cls) -> None:
        """Cleanup shared Playwright instance if no more users"""
        async with cls._ensure_loop_bound_lock():
            cls._shared_playwright_users -= 1
            if cls._shared_playwright_users == 0 and cls._shared_playwright is not None:
                await cls._shared_playwright.stop()
                cls._shared_playwright = None

    async def _setup_tracing(self) -> None:
        """Setup Playwright tracing if enabled in config"""
        if not self.config.tracing.enabled:
            self.logger.debug("Tracing not enabled")
            return

        # Use trace output path directly as file path
        self.trace_file_path = self.config.tracing.output_path

        # Start tracing with configured options
        await self.context.tracing.start(
            screenshots=self.config.tracing.get("screenshots", True),
            snapshots=self.config.tracing.get("snapshots", True),
            sources=self.config.tracing.get("sources", True),
        )
        self.logger.info(f"Tracing started, will save to: {self.trace_file_path}")

    async def _stop_tracing(self) -> None:
        """Stop tracing and save trace file"""
        if self.trace_file_path and self.context:
            try:
                await self.context.tracing.stop(path=self.trace_file_path)
                self.logger.info(f"Trace saved to: {self.trace_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save trace: {e}")
            finally:
                self.trace_file_path = None

    async def _start_recording(self) -> None:
        """Start screen recording using QuickRecorder"""
        try:
            self.logger.info("Starting screen recording setup...")

            # Create a new page for recording identification
            recording_page = await self.context.new_page()

            # Set the page title to the current UUID for window identification
            await recording_page.evaluate(
                f"""
                document.title = "{self.uuid}";
            """
            )

            # Give the page a moment to update the title
            await asyncio.sleep(2)

            # Use AppleScript to start QuickRecorder
            # Quickrecorder's recordings seem to be corrupted for chromium windows only
            # So far setting a lower quality and fps seem to fix it
            applescript = f"""
            tell application "QuickRecorder"
                activate
                configure {{fps:30, quality:2}}
                record window titled "{self.uuid}" in application "Chromium"
            end tell
            """

            # Execute AppleScript
            import subprocess

            result = subprocess.run(
                ["osascript", "-e", applescript],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                self.logger.info(
                    f"Successfully started recording for window with UUID: {self.uuid}"
                )
            else:
                self.logger.error(f"Failed to start recording: {result.stderr}")

            await asyncio.sleep(2)

            # Close the recording setup page
            await recording_page.close()

        except Exception as e:
            self.logger.error(f"Error starting recording: {e}")
            # Don't raise the exception - recording failure shouldn't stop the task

    async def _get_tabs_info(self) -> list[dict]:
        """Get information about all open tabs"""
        tabs_info = []
        for i, page in enumerate(self.context.pages):
            tabs_info.append(
                {
                    "id": i,
                    "title": await page.title(),
                    "url": page.url,
                    "is_active": page == self.page,
                }
            )
        return tabs_info

    async def setup(
        self,
        task_config: dict | None = None,
        headless: Optional[bool] = None,
    ):
        """Initialize the browser environment with configuration"""
        self.task_config = task_config
        self.context_manager = await self._ensure_playwright()

        # Get launch options from config and convert to dict
        launch_options = OmegaConf.to_container(
            self.config.browser.launch_options, resolve=True
        )

        if headless is not None:
            # update both the runtime options AND the config object,
            # so other code (like highlight_element’s “respect_headless”) reads the same value.
            launch_options["headless"] = bool(headless)
            try:
                self.config.browser.launch_options.headless = bool(headless)
            except Exception:
                pass

        # Add cache directory if configured
        if hasattr(self.config.browser, "cache_dir") and self.config.browser.cache_dir:
            # Use absolute path for cache directory
            cache_dir = Path(self.config.browser.cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            cache_arg = f"--disk-cache-dir={cache_dir}"
            launch_options["args"] = launch_options.get("args", []) + [cache_arg]
            self.logger.info(f"Browser cache configured: {cache_arg}")

        # Get context options from config and convert to dict
        context_options = OmegaConf.to_container(
            self.config.browser.context_options, resolve=True
        )

        # Check if user_data_dir is specified - use launch_persistent_context if so
        user_data_dir = None
        if (
            hasattr(self.config.browser, "user_data_dir")
            and self.config.browser.user_data_dir
        ):
            user_data_dir = self.config.browser.user_data_dir

        if user_data_dir:
            # Use launch_persistent_context for user data directory
            # Remove --disk-cache-dir from args since persistent context manages its own cache
            persistent_options = {**launch_options, **context_options}
            if "args" in persistent_options:
                persistent_options["args"] = [
                    arg
                    for arg in persistent_options["args"]
                    if not arg.startswith("--disk-cache-dir")
                ]

            self.context = (
                await self.context_manager.chromium.launch_persistent_context(
                    user_data_dir,
                    ignore_default_args=["--enable-automation"],
                    **persistent_options,
                )
            )
            self.browser = self.context.browser
            self.logger.info(
                f"Using persistent context with cache in user data dir: {user_data_dir}"
            )
        else:
            # Regular launch without persistent context
            self.browser = await self.context_manager.chromium.launch(**launch_options)
            self.context = await self.browser.new_context(**context_options)

        # Start tracing if enabled
        await self._setup_tracing()

        # Set default timeout for all locator actions
        self.context.set_default_timeout(self.config.browser.timeouts.default)

        # Add init script if it exists
        init_script_path = Path(self.config.init_script_path)
        if init_script_path.exists():
            with open(init_script_path) as f:
                await self.context.add_init_script(f.read())
        else:
            self.logger.warning(f"Init script not found: {init_script_path}")

        # Create initial page (or use existing one from persistent context)
        if self.context.pages:
            # Use existing page from persistent context
            self.page = self.context.pages[0]
        else:
            # Create new page for regular context
            self.page = await self.context.new_page()

        # Handle authentication before navigating to start_url
        await Stealth().apply_stealth_async(self.page)
        """
        if self.task_config and "sites" in self.task_config:
            required_sites = self.task_config["sites"]
            await self.ensure_logged_in(required_sites)
            """

        # Start recording if enabled
        if self.config.recording.enabled:
            await self._start_recording()

        # Navigate to start URL from task config
        if self.task_config and "start_url" in self.task_config:
            await self.page.goto(
                self.task_config["start_url"],
                wait_until="domcontentloaded",
                timeout=self.config.browser.timeouts.page_load_domcontent,
            )
            await self._maybe_bypass_storefront_password_gate()
        else:
            self.logger.warning("No start_url specified in task config")
        return await self.observation()

    async def new_tab(self, url: str | None = None) -> int:
        """Create a new tab and optionally navigate to URL. Returns tab ID."""
        async with ElementHighlight(sleep=0.5):
            page = await self.context.new_page()
            if url:
                await page.goto(url, wait_until="domcontentloaded")
            self.page = page  # Make new tab active
            return len(self.context.pages) - 1

    async def switch_tab(self, tab_id: int) -> None:
        """Switch to a different tab by ID"""
        async with ElementHighlight(sleep=0.3):
            if 0 <= tab_id < len(self.context.pages):
                self.page = self.context.pages[tab_id]
                await self.page.bring_to_front()
            else:
                raise ValueError(f"Invalid tab ID: {tab_id}")

    async def close_tab(self, tab_id: int) -> None:
        """Close a tab by ID"""
        async with ElementHighlight(sleep=0.3):
            if 0 <= tab_id < len(self.context.pages):
                page = self.context.pages[tab_id]
                await page.close()
                # If we closed the active tab, switch to the currently activated tab from context
                if page == self.page and self.context.pages:
                    # Find the currently active/focused tab in the context
                    for p in self.context.pages:
                        try:
                            if await p.evaluate("document.hasFocus()"):
                                self.page = p
                                break
                        except Exception:
                            continue
                    else:
                        # Fallback to last tab if no focused tab found
                        self.page = self.context.pages[-1]

                    # Ensure the new active page is brought to front
                    await self.page.bring_to_front()
            else:
                raise ValueError(f"Invalid tab ID: {tab_id}")

    async def screenshot(self) -> str:
        """
        Capture screenshot of the current tab and return the base64 encoded image.
        """
        screenshot_bytes = await self.page.screenshot(full_page=False)
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def reset(self):
        """Reset the environment to initial state"""
        # Close all tabs
        for page in self.context.pages:
            await page.close()
        self.page = await self.context.new_page()

        # Return to start URL from task config
        if self.task_config and "start_url" in self.task_config:
            await self.page.goto(
                self.task_config["start_url"], wait_until="domcontentloaded"
            )
        else:
            self.logger.warning("No start_url specified in task config")
        return await self.observation()

    async def _refresh_dom_tags(self) -> None:
        parser_script_path = Path(self.config.parser_script_path)
        if not parser_script_path.exists():
            return
        timeout_s = self.config.browser.timeouts.dom_tag_refresh / 1000
        try:
            parser_code = parser_script_path.read_text()
            await asyncio.wait_for(self.page.evaluate(parser_code), timeout=timeout_s)
        except (asyncio.TimeoutError, Exception) as e:
            self.logger.debug(f"DOM tag refresh skipped: {e}")

    async def step(
        self,
        action: str,
        before_hook: Callable[[], None] = None,
        after_hook: Callable[[], None] = None,
    ):
        """
        Execute an action in the environment using JSON string format and return the next observation.

        Args:
            action: JSON string describing the action to execute

        Returns:
            dict: The observation after executing the action (same format as observation() method)

        Examples:
            obs = await env.step('{"action": "click", "target": "login_button"}')
            obs = await env.step('{"action": "type", "target": "username", "text": "john_doe", "enter": true}')
            obs = await env.step('{"action": "select", "target": "country", "value": "US"}')
            obs = await env.step('{"action": "goto_url", "url": "https://example.com"}')
            obs = await env.step('{"action": "back"}')
            obs = await env.step('{"action": "new_tab", "url": "https://example.com"}')
            obs = await env.step('{"action": "switch_tab", "tab_id": 1}')
            obs = await env.step('{"action": "close_tab", "tab_id": 1}')
            obs = await env.step('{"action": "terminate", "answer": "The product costs $29.99"}')
        """
        import json

        try:
            action_data = json.loads(action)
            action_name = action_data.get("action")

            if action_name in (
                "click",
                "type",
                "hover",
                "select",
                "clear",
                "key_press",
            ) and self.config.get("refresh_dom_tags_per_action", False):
                await self._refresh_dom_tags()

            if action_name == "click":
                await self.click(action_data["target"])

            elif action_name == "mouse_click":
                await self.mouse_click(action_data["at_x"], action_data["at_y"])

            elif action_name == "type":
                text = action_data["text"]
                target = action_data["target"]
                press_enter = action_data.get("enter", False)
                await self.type(target, text, press_enter)

            elif action_name == "raw_type":
                await self.raw_type(action_data["text"])

            elif action_name == "scroll":
                await self.scroll(action_data["direction"], action_data["amount"])

            elif action_name == "hover":
                await self.hover(action_data["target"])

            elif action_name == "move":
                # Park the cursor at the viewport corner to dismiss
                # hover-persistent dropdown/flyout menus (fires mouseleave).
                await self.mouse_move(0, 0)

            elif action_name == "select":
                await self.select(action_data["target"], action_data["value"])

            elif action_name == "clear":
                await self.clear(action_data["target"])

            elif action_name == "key_press":
                key = action_data["key"]
                target = action_data.get("target")
                await self.key_press(key, target)

            elif action_name == "goto_url":
                await self.goto_url(action_data["url"])

            elif action_name == "back":
                await self.back()

            elif action_name == "forward":
                await self.forward()

            elif action_name == "refresh":
                await self.refresh()

            elif action_name == "new_tab":
                url = action_data.get("url")
                await self.new_tab(url)

            elif action_name == "switch_tab":
                tab_id = action_data["tab_id"]
                await self.switch_tab(tab_id)

            elif action_name == "close_tab":
                tab_id = action_data["tab_id"]
                await self.close_tab(tab_id)

            elif action_name == "terminate":
                answer = action_data.get("answer", "")
                await self.terminate(answer)

            else:
                self.logger.error(f"Unknown action: {action_name}")
                raise ValueError(f"Unknown action: {action_name}")

            # Sleep after action if configured
            if self.config.browser.sleep_after_action > 0:
                await asyncio.sleep(self.config.browser.sleep_after_action)

            # Return the next observation after executing the action
            observation = await self.observation()
            observation["error"] = None
            return observation

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON action format: {action}")
            observation = await self.observation()
            observation["error"] = f"Invalid JSON action format: {e}"
            return observation
        except KeyError as e:
            self.logger.error(f"Missing required parameter in action: {e}")
            observation = await self.observation()
            observation["error"] = f"Missing required parameter in action: {e}"
            return observation
        except Exception as e:
            self.logger.error(f"Error executing action: {action}, error: {e}")
            observation = await self.observation()
            observation["error"] = f"Error executing action: {e}"
            return observation

    # ===================================================================
    # ACTION METHODS
    # ===================================================================

    async def click(self, semantic_id: str) -> None:
        """
        Click on an element identified by its semantic ID.

        Args:
            semantic_id: The parser-semantic-id of the element to click

        Example:
            await env.click("login_button")
            await env.click("menu.settings")
        """
        async with ElementHighlight(semantic_id, sleep=0.5, center=False):
            selector = f'[parser-semantic-id="{semantic_id}"]'
            element = self.page.locator(selector)
            await element.click()
            self.logger.info(f"Clicked element: {semantic_id}")

    async def mouse_move(self, x: int, y: int) -> None:
        """
        Move the cursor/mouse to a target coordinate.

        Args:
            x: The x-coordinate to move the cursor/mouse
            y: The y-coordinate to move the cursor/mouse

        Example:
            await env.mouse_move(200, 100)
        """

        await self.page.mouse.move(x, y)

        self.logger.info(f"Moved mouse to [{x}, {y}]")

    async def mouse_click(self, at_x: int = None, at_y: int = None) -> None:
        """
        Perform a mouse click on the given coordinates. Defaults to the
        current mouse position if coordinates not given.

        Example:
            await env.mouse_click()
            await env.mouse_click(200, 100)
        """

        await self.page.mouse.click(x=at_x, y=at_y)

        self.logger.info(f"Performed a raw click at [{at_x}, {at_y}]")

    async def type(
        self, semantic_id: str, text: str, press_enter: bool = False
    ) -> None:
        """
        Type text into an input element.

        Args:
            semantic_id: The parser-semantic-id of the input element
            text: Text to type
            press_enter: Whether to press Enter after typing

        Example:
            await env.type("search_input", "hello world")
            await env.type("username", "john_doe", press_enter=True)
        """
        async with ElementHighlight(semantic_id, sleep=0.5, center=True):
            selector = f'[parser-semantic-id="{semantic_id}"]'
            element = self.page.locator(selector)

            # Short timeout scroll - fail fast on hallucinated elements
            await element.scroll_into_view_if_needed(timeout=500)
            await element.fill(text, force=True)  # Clear and type

            if press_enter:
                await element.press("Enter")

            self.logger.info(f"Typed '{text}' into element: {semantic_id}")

    async def raw_type(self, text: str):
        """
        Types the given string into the page as raw keyboard input.
        Will do nothing if an input element is not selected

        Args:
            text: Text to type

        Example:
            await env.raw_type("Best star wars movies")
        """
        await self.page.keyboard.type(text, delay=50)

        self.logger.info(f"Typed '{text}' into the page as raw keyboard input")

    async def scroll(self, direction: str, amount: int) -> None:
        """
        Performs a mouse scroll at the page inside the viewport.

        Args:
            direction: "up", "down", "left", or "right". The direction to scroll by.
            amount: The amount of times to perform the scroll in that direction.
        """
        if direction not in ["up", "down", "left", "right"]:
            raise ValueError(f"Invalid direction: {direction}")

        scroll_button = {
            "up": "ArrowUp",
            "down": "ArrowDown",
            "left": "ArrowLeft",
            "right": "ArrowRight",
        }[direction]

        for _ in range(amount * 3):  # one mouse scroll event ~= 3 arrow keys
            await self.page.keyboard.press(scroll_button, delay=50)

        self.logger.info(f"Scrolled the page {direction} by {amount} times")

    async def hover(self, semantic_id: str) -> None:
        """
        Hover over an element to trigger tooltips or dropdown menus.

        Args:
            semantic_id: The parser-semantic-id of the element to hover over

        Example:
            await env.hover("menu_item")
            await env.hover("tooltip_trigger")
        """
        async with ElementHighlight(semantic_id, sleep=0.5, center=True):
            selector = f'[parser-semantic-id="{semantic_id}"]'
            element = self.page.locator(selector)

            # Short timeout scroll - fail fast on hallucinated elements
            await element.scroll_into_view_if_needed(timeout=500)
            await element.hover(force=True)
            self.logger.info(f"Hovered over element: {semantic_id}")

    async def select(self, semantic_id: str, value: str) -> None:
        """
        Select an option from a dropdown/select element.

        Args:
            semantic_id: The parser-semantic-id of the select element
            value: The value of the option to select

        Example:
            await env.select("country_dropdown", "USA")
            await env.select("language_select", "en")
        """
        async with ElementHighlight(semantic_id, sleep=0.5, center=True):
            selector = f'[parser-semantic-id="{semantic_id}"]'
            element = self.page.locator(selector)

            # Short timeout scroll - fail fast on hallucinated elements
            await element.scroll_into_view_if_needed(timeout=500)
            await element.select_option(value, force=True)
            self.logger.info(f"Selected '{value}' in element: {semantic_id}")

    async def clear(self, semantic_id: str) -> None:
        """
        Clear the content of an input element.

        Args:
            semantic_id: The parser-semantic-id of the input element to clear

        Example:
            await env.clear("search_input")
            await env.clear("comment_textarea")
        """
        async with ElementHighlight(semantic_id, sleep=0.5, center=True):
            selector = f'[parser-semantic-id="{semantic_id}"]'
            element = self.page.locator(selector)

            # Short timeout scroll - fail fast on hallucinated elements
            await element.scroll_into_view_if_needed(timeout=500)
            await element.clear(force=True)
            self.logger.info(f"Cleared element: {semantic_id}")

    async def key_press(self, key: str, semantic_id: str | None = None) -> None:
        """
        Press a keyboard key, optionally on a specific element.

        Args:
            key: Key to press (e.g., "Enter", "Escape", "Tab", "ArrowDown")
            semantic_id: Optional element to focus before pressing key

        Example:
            await env.key_press("Escape")  # Press Escape globally
            await env.key_press("Enter", "search_input")  # Press Enter on search input
            await env.key_press("ArrowDown", "dropdown")  # Navigate dropdown
        """
        async with ElementHighlight(semantic_id, sleep=0.3, center=True):
            if semantic_id:
                selector = f'[parser-semantic-id="{semantic_id}"]'
                element = self.page.locator(selector)
                # Short timeout scroll - fail fast on hallucinated elements
                await element.scroll_into_view_if_needed(timeout=500)
                await element.press(key, force=True)
                self.logger.info(f"Pressed '{key}' on element: {semantic_id}")
            else:
                await self.page.keyboard.press(key)
                self.logger.info(f"Pressed '{key}' globally")

    # ===================================================================
    # NAVIGATION ACTIONS
    # ===================================================================

    async def goto_url(self, url: str) -> None:
        """
        Navigate to a specific URL in the current tab.

        Args:
            url: URL to navigate to

        Example:
            await env.goto_url("https://google.com")
            await env.goto_url("http://localhost:3000/login")
        """
        async with ElementHighlight(sleep=0.5):
            await self.page.goto(url, wait_until="domcontentloaded")
            self.logger.info(f"Navigated to: {url}")

    async def back(self) -> None:
        """
        Navigate back in browser history.

        Example:
            await env.back()
        """
        async with ElementHighlight(sleep=0.5):
            await self.page.go_back(wait_until="domcontentloaded")
            self.logger.info("Navigated back")

    async def forward(self) -> None:
        """
        Navigate forward in browser history.

        Example:
            await env.forward()
        """
        async with ElementHighlight(sleep=0.5):
            await self.page.go_forward(wait_until="domcontentloaded")
            self.logger.info("Navigated forward")

    async def refresh(self) -> None:
        """
        Refresh/reload the current page.

        Example:
            await env.refresh()
        """
        async with ElementHighlight(sleep=0.5):
            await self.page.reload(wait_until="domcontentloaded")
            self.logger.info("Page refreshed")

    async def terminate(self, answer: str = "") -> None:
        """
        Terminate the task with an optional answer.

        Args:
            answer: The model's final answer/response for the task

        Example:
            await env.terminate("The product costs $29.99")
            await env.terminate()  # Terminate without answer
        """
        async with ElementHighlight(sleep=0.3):
            self.model_answer = answer
            if answer:
                self.logger.info(f"Task terminated with answer: {answer}")
            else:
                self.logger.info("Task terminated without answer")

    async def _wait_for_custom_network_idle(
        self, timeout_ms: int = 10000, idle_time_ms: int = 500
    ) -> None:
        """
        Custom network idle detection that works with XHR/fetch requests.
        Uses async JavaScript Promise-based waiting for better performance.
        """
        self.logger.info(
            f"Waiting for custom network idle (timeout: {timeout_ms}ms, idle: {idle_time_ms}ms)"
        )

        try:
            # Add Python-side timeout as a safety net
            timeout_future = asyncio.create_task(asyncio.sleep(timeout_ms / 1000))
            evaluate_future = asyncio.create_task(
                self.page.evaluate(
                    """
                async ([idleTimeMs, timeoutMs]) => {
                    if (typeof window.__networkActivity === 'undefined') {
                        console.log('Network activity tracker not available');
                        return true; // Fallback if tracker not available
                    }

                    console.log('Starting network idle wait...');
                    try {
                        const isIdle = await window.__networkActivity.waitForIdle(idleTimeMs, timeoutMs);
                        console.log('Network idle wait completed:', isIdle);
                        return isIdle;
                    } catch (error) {
                        console.warn('Network idle wait error:', error);
                        return false;
                    }
                }
            """,
                    [idle_time_ms, timeout_ms],
                )
            )

            # Race between evaluation and timeout
            done, pending = await asyncio.wait(
                [evaluate_future, timeout_future], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            if evaluate_future in done:
                result = await evaluate_future
                if result:
                    self.logger.info("Custom network idle detected")
                else:
                    self.logger.warning(
                        f"Custom network idle timeout after {timeout_ms}ms"
                    )
            else:
                self.logger.warning(
                    "Custom network idle detection timed out on Python side"
                )

        except Exception as e:
            self.logger.warning(f"Custom network idle check failed: {e}")
            # Fallback to old polling method
            await self._wait_for_custom_network_idle_fallback(timeout_ms, idle_time_ms)

    async def _wait_for_custom_network_idle_fallback(
        self, timeout_ms: int = 10000, idle_time_ms: int = 500
    ) -> None:
        """
        Fallback polling-based network idle detection.
        """
        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout_ms / 1000

        self.logger.info("Using fallback network idle detection")

        while True:
            try:
                # Check if our network tracker is available and if network is idle
                is_idle = await self.page.evaluate(
                    """
                    (idleTimeMs) => {
                        if (typeof window.__networkActivity === 'undefined') {
                            return true; // Fallback if tracker not available
                        }
                        return window.__networkActivity.isIdle(idleTimeMs);
                    }
                """,
                    idle_time_ms,
                )

                if is_idle:
                    self.logger.info("Custom network idle detected (fallback)")
                    break

                # Check timeout
                if (asyncio.get_event_loop().time() - start_time) >= timeout_seconds:
                    self.logger.warning(
                        f"Custom network idle timeout after {timeout_ms}ms (fallback)"
                    )
                    break

                # Wait a bit before checking again
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.warning(f"Custom network idle fallback check failed: {e}")
                break

    async def observation(self):
        """Get parsed page content using the parser script"""
        parser_script_path = Path(self.config.parser_script_path)
        content = {}

        # Wait for page to be fully loaded and stable
        try:
            self.logger.info("Waiting for page to be fully loaded and stable")
            await self.page.wait_for_load_state(
                "domcontentloaded",
                timeout=self.config.browser.timeouts.page_load_domcontent,
            )

            # Use both original networkidle (for page loads) and custom detection (for XHR/fetch)
            try:
                # First wait for Playwright's networkidle (handles initial page loads well)
                await self.page.wait_for_load_state(
                    "networkidle",
                    timeout=self.config.browser.timeouts.page_load_networkidle,
                )  # Shorter timeout
                self.logger.info("Playwright networkidle detected")
            except Exception as e:
                self.logger.info(f"Playwright networkidle timeout (normal): {e}")

            # Then wait for custom network idle detection (handles XHR/fetch after interactions)
            await self._wait_for_custom_network_idle(
                timeout_ms=self.config.browser.timeouts.page_load_networkidle,
                idle_time_ms=self.config.browser.timeouts.custom_network_idle,
            )
            if self.wait_hook:
                await self.wait_hook(self.page)

            self.logger.info("Page loaded and stable")
        except Exception as e:
            self.logger.warning(f"Page load wait timeout: {e}")

        # Additional safety check - wait for body element
        # try:
        #     await self.page.wait_for_selector(
        #         "body", timeout=self.config.browser.timeouts.element_wait
        #     )
        # except Exception as e:
        #     self.logger.warning(f"Body element not found: {e}")

        if parser_script_path.exists():
            with open(parser_script_path) as f:
                parser_code = f.read()
            parser_timeout_s = self.config.browser.timeouts.parser_eval / 1000
            try:
                content = await asyncio.wait_for(
                    self.page.evaluate(parser_code), timeout=parser_timeout_s
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Parser script timed out after {parser_timeout_s}s; falling back to raw HTML"
                )
                content = {"html": await self.page.content()}
            except Exception as e:
                self.logger.error(f"Parser script failed: {e}")
                content = {"html": await self.page.content()}
        else:
            self.logger.warning(f"Parser script not found: {parser_script_path}")
            content = {"html": await self.page.content()}

        # Keep the observation contract stable when parser.js is missing, times out,
        # or fails. Agents can still inspect the raw HTML and will see that there are
        # no parser-generated semantic targets instead of failing with a KeyError.
        content.setdefault("url_map", {})
        content.setdefault("clickable_elements", [])
        content.setdefault("hoverable_elements", [])
        content.setdefault("input_elements", [])
        content.setdefault("select_elements", [])

        # Add tabs information to the observation
        content["tabs"] = await self._get_tabs_info()

        # Add model answer if available
        content["model_answer"] = self.model_answer

        # Add evaluation information
        if (
            self.task_config
            and "eval" in self.task_config
            and self.config.get("evaluation", {}).get("enabled", True)
        ):
            score = await self.evaluate_task()
            content["score"] = score

            # Always terminate if model called terminate
            content["terminated"] = self.model_answer is not None or score != 0.0
        else:
            content["score"] = 0.0
            content["terminated"] = self.model_answer is not None

        return content

    async def close(self):
        """Clean up pages/contexts/browsers deterministically, then shrink PW refcount."""
        # 1) Stop tracing/recording scoped to this context
        with contextlib.suppress(Exception):
            await self._stop_tracing()

        # 2) Close per-env resources (these are always safe to close)
        with contextlib.suppress(Exception):
            if self.page:
                await self.page.close()
        with contextlib.suppress(Exception):
            if self.context:
                await self.context.close()

        # 3) Close this env's browser (you launch a new browser per env in setup)
        with contextlib.suppress(Exception):
            if self.browser:
                await self.browser.close()

        # 4) Null out handles (helps debugging and double-close safety)
        self.page = None
        self.context = None
        self.browser = None

        # 5) Decrement the shared Playwright user count; stop PW only when last user is gone
        #    (stopping PW early would nuke other running envs)
        with contextlib.suppress(Exception):
            if self.context_manager:  # you use this as the "shared PW" handle
                await self._cleanup_playwright()
                self.context_manager = None

    async def _delete_container_with_retry(
        self,
        incus_server_url: str,
        container_name: str,
        proxy_server: str | None,
        max_retries: int = 2,
    ) -> bool:
        """
        Delete a container with retry logic.

        Args:
            incus_server_url: Incus server URL
            container_name: Name of container to delete
            proxy_server: Optional proxy server URL for HTTP requests
            max_retries: Maximum number of retry attempts

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        from rl_web_agent.incus_client import delete_container

        for attempt in range(max_retries + 1):
            try:
                await delete_container(
                    incus_server_url, container_name, proxy_server=proxy_server
                )
                self.logger.info(f"🗑️ Deleted container {container_name}")
                return True
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(
                        f"⚠️ Failed to delete {container_name} (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(1)  # Wait 1 second before retry
                else:
                    self.logger.error(
                        f"❌ Final failure deleting {container_name}: {e}"
                    )
                    return False
        return False

    # ===================================================================
    # DEBUG HELPERS
    # ===================================================================

    def debug_pause(self):
        input(
            "[DEBUG] Environment paused for manual page navigation, log in, etc. Press ENTER to continue."
        )
