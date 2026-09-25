"""Plain-python checks for driving a vLLM-style engine while reducing outputs as they finish.

Run: PYTHONPATH=src uv run python tests/test_engine_streaming.py
Logprob-enabled RequestOutputs are large, so they must be reduced the moment they
finish instead of being retained until the whole dataset is done. All prompts are
still submitted in ONE batch, so there is a single progress bar and self-consistency
copies of a prompt are never split across submissions.
"""
import gc
import sys
import weakref
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.engine_streaming import generate_streaming


class _FakeOutput:
    def __init__(self, request_id: str, finished: bool = True) -> None:
        self.request_id: str = request_id
        self.finished: bool = finished


class _FakeRenderer:
    """Renders prompt dicts into engine inputs, like vLLM's ``Renderer.render_cmpl``."""

    def __init__(self, calls: list[str], fail: bool = False) -> None:
        self.calls: list[str] = calls
        self.fail: bool = fail
        self.rendered: list[list[dict[str, str]]] = []

    def render_cmpl(self, prompts: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.fail:
            raise ValueError("prompt too long for the model")
        self.calls.append("render")
        self.rendered.append(list(prompts))
        return [{"type": "rendered", "text": prompt["prompt"]} for prompt in prompts]


class _FakeEngine:
    """Finishes requests in a scripted order, one per step, like a vLLM LLMEngine."""

    def __init__(
        self,
        finish_order: list[int] | None = None,
        fail_on_add: int | None = None,
        emit_unfinished: bool = False,
        fail_on_render: bool = False,
    ) -> None:
        self.finish_order: list[int] | None = finish_order
        self.fail_on_add: int | None = fail_on_add
        self.emit_unfinished: bool = emit_unfinished
        self.calls: list[str] = []
        self.renderer: _FakeRenderer = _FakeRenderer(self.calls, fail=fail_on_render)
        self.added: list[tuple[str, Any, Any]] = []
        self.aborted: list[tuple[list[str], bool]] = []
        self.steps_done: int = 0
        self.finished_at_step: dict[str, int] = {}
        self.output_refs: list[weakref.ref[_FakeOutput]] = []
        self._queue: list[str] = []

    def add_request(self, request_id: str, prompt: Any, params: Any) -> str:
        if self.fail_on_add is not None and len(self.added) == self.fail_on_add:
            raise RuntimeError("prompt too long")
        self.calls.append("add")
        self.added.append((request_id, prompt, params))
        self._queue.append(request_id)
        # vLLM returns an internal id that differs from the one passed in.
        return f"{request_id}-internal"

    def has_unfinished_requests(self) -> bool:
        return bool(self._queue)

    def step(self) -> list[_FakeOutput]:
        self.calls.append("step")
        self.steps_done += 1
        if self.finish_order is not None:
            request_id = str(self.finish_order[self.steps_done - 1])
            self._queue.remove(request_id)
        else:
            request_id = self._queue.pop(0)
        self.finished_at_step[request_id] = self.steps_done
        outputs: list[_FakeOutput] = [_FakeOutput(request_id)]
        if self.emit_unfinished and self._queue:
            outputs.insert(0, _FakeOutput(self._queue[0], finished=False))
        self.output_refs.extend(weakref.ref(output) for output in outputs)
        return outputs

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        self.aborted.append((list(request_ids), internal))


def _reduce(output: _FakeOutput) -> str:
    return f"result-{output.request_id}"


def test_all_prompts_are_submitted_in_one_batch_before_stepping() -> None:
    engine = _FakeEngine()
    generate_streaming(engine, ["a", "b", "c"], ["pa", "pb", "pc"], _reduce)
    assert engine.calls == ["render", "add", "add", "add", "step", "step", "step"], engine.calls
    print("test_all_prompts_are_submitted_in_one_batch_before_stepping PASSED")


def test_prompts_are_rendered_in_one_call_and_submitted_as_engine_inputs() -> None:
    # vLLM deprecated passing raw prompt strings to add_request: they must go
    # through Renderer.render_cmpl first, and the rendered inputs get submitted.
    engine = _FakeEngine()
    generate_streaming(engine, ["a", "b", "c"], ["pa", "pb", "pc"], _reduce)
    assert engine.renderer.rendered == [
        [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    ], engine.renderer.rendered
    assert [(prompt, params) for _, prompt, params in engine.added] == [
        ({"type": "rendered", "text": "a"}, "pa"),
        ({"type": "rendered", "text": "b"}, "pb"),
        ({"type": "rendered", "text": "c"}, "pc"),
    ], engine.added
    print("test_prompts_are_rendered_in_one_call_and_submitted_as_engine_inputs PASSED")


def test_failed_rendering_submits_nothing() -> None:
    engine = _FakeEngine(fail_on_render=True)
    try:
        generate_streaming(engine, ["a", "b"], ["p"] * 2, _reduce)
    except ValueError as error:
        assert "prompt too long" in str(error)
    else:
        raise AssertionError("expected the render failure to propagate")
    assert engine.added == [] and engine.aborted == [], (engine.added, engine.aborted)
    print("test_failed_rendering_submits_nothing PASSED")


def test_results_follow_prompt_order_when_finishing_out_of_order() -> None:
    engine = _FakeEngine(finish_order=[2, 0, 1])
    results = generate_streaming(engine, ["a", "b", "c"], ["p"] * 3, _reduce)
    assert results == ["result-0", "result-1", "result-2"], results
    print("test_results_follow_prompt_order_when_finishing_out_of_order PASSED")


def test_outputs_are_reduced_in_the_step_they_finish() -> None:
    engine = _FakeEngine(finish_order=[1, 2, 0])
    reduced_at_step: dict[str, int] = {}

    def reduce_and_record(output: _FakeOutput) -> str:
        reduced_at_step[output.request_id] = engine.steps_done
        return _reduce(output)

    generate_streaming(engine, ["a", "b", "c"], ["p"] * 3, reduce_and_record)
    assert reduced_at_step == engine.finished_at_step, (reduced_at_step, engine.finished_at_step)
    print("test_outputs_are_reduced_in_the_step_they_finish PASSED")


def test_raw_outputs_are_not_retained() -> None:
    engine = _FakeEngine()
    alive_when_reduced: list[int] = []

    def reduce_and_count_alive(output: _FakeOutput) -> str:
        gc.collect()
        alive_when_reduced.append(sum(ref() is not None for ref in engine.output_refs))
        return _reduce(output)

    results = generate_streaming(engine, list("abcdef"), ["p"] * 6, reduce_and_count_alive)
    # Only the output being reduced may be alive: earlier ones were already dropped.
    assert alive_when_reduced == [1] * 6, alive_when_reduced
    gc.collect()
    assert all(ref() is None for ref in engine.output_refs)
    assert len(results) == 6
    print("test_raw_outputs_are_not_retained PASSED")


def test_unfinished_outputs_are_skipped() -> None:
    engine = _FakeEngine(emit_unfinished=True)
    reduced: list[str] = []

    def reduce_and_record(output: _FakeOutput) -> str:
        reduced.append(output.request_id)
        return _reduce(output)

    results = generate_streaming(engine, ["a", "b", "c"], ["p"] * 3, reduce_and_record)
    assert reduced == ["0", "1", "2"], reduced
    assert results == ["result-0", "result-1", "result-2"], results
    print("test_unfinished_outputs_are_skipped PASSED")


def test_on_finished_fires_once_per_prompt_for_a_single_progress_bar() -> None:
    engine = _FakeEngine(finish_order=[1, 0, 2])
    seen: list[str] = []
    generate_streaming(
        engine,
        ["a", "b", "c"],
        ["p"] * 3,
        _reduce,
        on_finished=lambda output: seen.append(output.request_id),
    )
    assert seen == ["1", "0", "2"], seen
    print("test_on_finished_fires_once_per_prompt_for_a_single_progress_bar PASSED")


def test_failed_submission_aborts_already_added_requests() -> None:
    engine = _FakeEngine(fail_on_add=2)
    try:
        generate_streaming(engine, ["a", "b", "c"], ["p"] * 3, _reduce)
    except RuntimeError as error:
        assert "prompt too long" in str(error)
    else:
        raise AssertionError("expected the add_request failure to propagate")
    assert engine.aborted == [(["0-internal", "1-internal"], True)], engine.aborted
    assert "step" not in engine.calls
    print("test_failed_submission_aborts_already_added_requests PASSED")


def test_failed_reduction_aborts_requests_still_in_flight() -> None:
    engine = _FakeEngine()

    def reduce_or_fail(output: _FakeOutput) -> str:
        if output.request_id == "1":
            raise ValueError("bad output")
        return _reduce(output)

    try:
        generate_streaming(engine, ["a", "b", "c"], ["p"] * 3, reduce_or_fail)
    except ValueError:
        pass
    else:
        raise AssertionError("expected the reducer failure to propagate")
    # Request 0 finished cleanly; 1 failed while reducing and 2 never ran.
    assert engine.aborted == [(["1-internal", "2-internal"], True)], engine.aborted
    print("test_failed_reduction_aborts_requests_still_in_flight PASSED")


def test_mismatched_lengths_are_rejected() -> None:
    try:
        generate_streaming(_FakeEngine(), ["a", "b"], ["p"], _reduce)
    except ValueError:
        print("test_mismatched_lengths_are_rejected PASSED")
        return
    raise AssertionError("expected ValueError for mismatched prompts/params")


def test_empty_prompts_do_not_touch_the_engine() -> None:
    engine = _FakeEngine()
    assert generate_streaming(engine, [], [], _reduce) == []
    assert engine.calls == []
    print("test_empty_prompts_do_not_touch_the_engine PASSED")


if __name__ == "__main__":
    test_all_prompts_are_submitted_in_one_batch_before_stepping()
    test_prompts_are_rendered_in_one_call_and_submitted_as_engine_inputs()
    test_failed_rendering_submits_nothing()
    test_results_follow_prompt_order_when_finishing_out_of_order()
    test_outputs_are_reduced_in_the_step_they_finish()
    test_raw_outputs_are_not_retained()
    test_unfinished_outputs_are_skipped()
    test_on_finished_fires_once_per_prompt_for_a_single_progress_bar()
    test_failed_submission_aborts_already_added_requests()
    test_failed_reduction_aborts_requests_still_in_flight()
    test_mismatched_lengths_are_rejected()
    test_empty_prompts_do_not_touch_the_engine()
    print("ALL PASSED")
