"""Regression tests for EOS removal from rollout responses."""

import argparse
from argparse import Namespace


def test_rollout_defaults_strip_matched_eos(monkeypatch):
    """A completed Qwen response must not include ``<|im_end|>`` in its text."""
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", "/tmp/slime-rollout-stop-test")

    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

    from slime.rollout import sglang_rollout
    from slime.utils.arguments import get_slime_extra_args_provider

    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)
    assert parser.parse_args(["--rollout-batch-size", "1"]).rollout_skip_special_tokens is True

    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(sglang_rollout, "get_rollout_num_engines", lambda args: 1)

    args = Namespace(
        hf_checkpoint="/unused",
        sglang_server_concurrency=1,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=128,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        sglang_enable_deterministic_inference=False,
        sglang_dp_size=1,
    )
    sglang_rollout.GenerateState.clear_instances()
    try:
        state = sglang_rollout.GenerateState(args)
        assert state.sampling_params["skip_special_tokens"] is True
        assert state.sampling_params["no_stop_trim"] is False

        response = "def answer():\n    return 42<|im_end|>"
        trimmed = DetokenizerManager.trim_matched_stop(
            None, response, {"matched": "<|im_end|>"}, state.sampling_params["no_stop_trim"]
        )
        assert trimmed == "def answer():\n    return 42"
        compile(trimmed, "<rollout>", "exec")
    finally:
        sglang_rollout.GenerateState.clear_instances()
