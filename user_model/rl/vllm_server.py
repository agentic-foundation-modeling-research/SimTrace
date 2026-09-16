"""Start TRL's vLLM rollout server with CUDA-safe multiprocessing.

The ``trl`` console entry point can import modules that inspect CUDA before the
vLLM worker pool is created. On Linux, inheriting that initialized CUDA runtime
through the default ``fork`` context fails. This wrapper selects ``spawn``
before importing any TRL, PyTorch, or vLLM modules.
"""

from __future__ import annotations

import multiprocessing as mp
import os


# Both controls are intentional: vLLM consults its environment variable for
# worker processes, while libraries that use Python multiprocessing directly
# inherit the process-wide start method.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
mp.set_start_method("spawn", force=True)


def main() -> None:
    """Parse the standard TRL server arguments and start the rollout server."""
    # Import only after spawn has been selected so no parent-side CUDA probe can
    # poison child worker initialization.
    from trl.scripts.vllm_serve import main as serve
    from trl.scripts.vllm_serve import make_parser

    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    serve(script_args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
