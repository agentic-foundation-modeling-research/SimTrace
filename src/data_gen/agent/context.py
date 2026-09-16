from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import LogApiCall

run_path = ContextVar("run_path", default=None)
api_call_manager: ContextVar[Optional["LogApiCall"]] = ContextVar(
    "api_call_manager", default=None
)
browser_context = ContextVar("browser_context", default=None)
