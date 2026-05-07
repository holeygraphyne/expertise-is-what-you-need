"""Tests for traces.calibration.cache."""
from traces.calibration.cache import cache_key


class TestCacheKey:
    def test_deterministic(self):
        k1 = cache_key("IS-foo", "openai/gpt-5.4", "A response.")
        k2 = cache_key("IS-foo", "openai/gpt-5.4", "A response.")
        assert k1 == k2

    def test_length_16(self):
        k = cache_key("IS-foo", "openai/gpt-5.4", "A response.")
        assert len(k) == 16

    def test_response_whitespace_changes_key(self):
        k1 = cache_key("IS-foo", "m", "Hello world")
        k2 = cache_key("IS-foo", "m", "Hello  world")  # two spaces
        k3 = cache_key("IS-foo", "m", "Hello world ")  # trailing
        assert k1 != k2
        assert k1 != k3

    def test_probe_change_changes_key(self):
        k1 = cache_key("IS-a", "m", "resp")
        k2 = cache_key("IS-b", "m", "resp")
        assert k1 != k2

    def test_model_change_changes_key(self):
        k1 = cache_key("IS-a", "m1", "resp")
        k2 = cache_key("IS-a", "m2", "resp")
        assert k1 != k2

    def test_is_hex(self):
        k = cache_key("IS-a", "m", "resp")
        int(k, 16)  # raises if not hex
