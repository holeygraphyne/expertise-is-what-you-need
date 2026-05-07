"""IS scorer calibration: two-stage LLM-as-judge audit tool.

See docs/superpowers/specs/2026-04-27-auditor-reframe-design.md
for design. Entry points:
  - `python -m traces calibrate judge`     Stage 1
  - `python -m traces calibrate recommend` Stage 2
"""
