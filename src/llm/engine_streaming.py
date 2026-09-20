"""Torch-free driver that runs a vLLM-style engine and reduces outputs as they finish.

``LLM.generate`` keeps every finished RequestOutput until the whole call returns.
With top-k logprobs that is several KB of host RAM per generated token, so a
dataset of self-consistency draws does not fit in memory. This mirrors vLLM's own
``LLM._run_engine`` loop but hands each finished output to a reducer immediately.
All prompts are still submitted up front, so scheduling is identical to
``LLM.generate``: one continuous batch, identical prompts stay adjacent for prefix
caching, and the caller can show a single progress bar.
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol


class IStreamingEngine(Protocol):
    """Subset of ``vllm.LLMEngine`` needed to drive generation step by step."""

    def add_request(self, request_id: str, prompt: str, params: Any) -> str: ...

    def step(self) -> list[Any]: ...

    def has_unfinished_requests(self) -> bool: ...

    def abort_request(self, request_ids: list[str], internal: bool = ...) -> None: ...


def generate_streaming[ResultT](
    engine: IStreamingEngine,
    prompts: Sequence[str],
    params: Sequence[Any],
    reduce_output: Callable[[Any], ResultT],
    on_finished: Callable[[Any], None] | None = None,
) -> list[ResultT]:
    """Generate for every prompt and return the reduced results in prompt order.

    Args:
        engine: vLLM-style engine with no other requests in flight.
        prompts: Formatted prompts, submitted in one batch.
        params: One sampling-params object per prompt.
        reduce_output: Turns a finished engine output into the value to keep; the
            raw output is dropped right after.
        on_finished: Optional hook called with each finished output (progress bar).

    Raises:
        ValueError: If ``prompts`` and ``params`` differ in length.
    """
    if len(prompts) != len(params):
        raise ValueError("prompts and params must have same length")
    if not prompts:
        return []

    # The engine reports outputs under the id passed in, but aborting needs the
    # internal id that add_request returns.
    internal_ids: dict[str, str] = {}
    results: dict[int, ResultT] = {}
    try:
        for index, (prompt, prompt_params) in enumerate(zip(prompts, params)):
            request_id: str = str(index)
            internal_ids[request_id] = engine.add_request(request_id, prompt, prompt_params)

        while engine.has_unfinished_requests():
            for output in engine.step():
                if not output.finished or output.request_id not in internal_ids:
                    continue
                results[int(output.request_id)] = reduce_output(output)
                if on_finished is not None:
                    on_finished(output)
                del internal_ids[output.request_id]
    except BaseException:
        if internal_ids:
            engine.abort_request(list(internal_ids.values()), internal=True)
        raise

    return [results[index] for index in range(len(prompts))]
