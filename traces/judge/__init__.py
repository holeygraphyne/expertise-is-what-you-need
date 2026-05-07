"""Blind LLM-as-a-judge path over benchmark outputs.

Distinct from `traces.calibration` (which does scorer-calibration audit).
This package implements blind, full-coverage judging with a multi-judge
panel, reporting `IFR-judge` alongside `IFR-det`.

See docs/superpowers/specs/2026-05-05-llm-judge-scorer-native-refactor.md
for the current scorer-native contract.
"""
