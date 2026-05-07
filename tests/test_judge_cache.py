"""Tests for traces.judge.cache."""
import json

import pytest

from traces.judge.cache import (
    cache_key_for,
    cache_key_has_case_hash,
    load_judge_cache,
    save_judge_cache,
)


class TestCacheKey:
    def test_deterministic(self):
        k1 = cache_key_for(
            probe_id="IS-bem", model="openai/gpt-5",
            response_sha256="a" * 64, judge_version="v1",
        )
        k2 = cache_key_for(
            probe_id="IS-bem", model="openai/gpt-5",
            response_sha256="a" * 64, judge_version="v1",
        )
        assert k1 == k2

    def test_components_in_key(self):
        k = cache_key_for(
            probe_id="IS-x", model="m", response_sha256="0" * 64, judge_version="v",
        )
        assert "IS-x" in k and "m" in k and "0" * 64 in k and "v" in k

    def test_case_payload_hash_changes_key(self):
        k1 = cache_key_for(
            probe_id="IS-x", model="m", response_sha256="0" * 64,
            case_sha256="a" * 64, judge_version="v",
        )
        k2 = cache_key_for(
            probe_id="IS-x", model="m", response_sha256="0" * 64,
            case_sha256="b" * 64, judge_version="v",
        )
        assert k1 != k2
        assert "a" * 64 in k1

    def test_current_key_shape_requires_case_hash(self):
        old_key = cache_key_for(
            probe_id="IS-x", model="m", response_sha256="0" * 64,
            judge_version="v",
        )
        new_key = cache_key_for(
            probe_id="IS-x", model="m", response_sha256="0" * 64,
            case_sha256="a" * 64, judge_version="v",
        )
        assert cache_key_has_case_hash(old_key) is False
        assert cache_key_has_case_hash(new_key) is True


class TestLoadSaveJudgeCache:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "judge_verdicts.json"
        data = {"key1": {"panel_verdict": {"x": 1}}}
        save_judge_cache(path, data)
        assert load_judge_cache(path) == data

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_judge_cache(tmp_path / "missing.json") == {}

    def test_corrupt_json_returns_empty_warns(self, tmp_path, caplog):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        with caplog.at_level("WARNING"):
            assert load_judge_cache(path) == {}
        assert "malformed" in caplog.text.lower()

    def test_errored_rows_filtered_on_load(self, tmp_path):
        path = tmp_path / "judge_verdicts.json"
        # Mix of healthy and errored rows.
        data = {
            "good_key": {"panel_verdict": {"_v": 1}, "_judge_version": "v1"},
            "bad_key":  {"error": "transient http 500"},
        }
        path.write_text(json.dumps(data))
        loaded = load_judge_cache(path)
        assert "good_key" in loaded
        assert "bad_key" not in loaded

    def test_atomic_write_no_partial_file(self, tmp_path):
        # Verify the .tmp file is cleaned up after a successful write.
        path = tmp_path / "judge_verdicts.json"
        save_judge_cache(path, {"k": {"x": 1}})
        assert path.exists()
        assert not (tmp_path / "judge_verdicts.json.tmp").exists()
