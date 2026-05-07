"""Shared YAML serialization for corpus paper files.

Block-scalar style is preserved for multi-line strings so hand-edited
preamble/central_claim/operational_request fields read naturally in
the YAML files we commit.
"""
from __future__ import annotations

from typing import Any

import yaml


class _BlockStyleDumper(yaml.SafeDumper):
    pass


def _represent_multiline_str(dumper: Any, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_BlockStyleDumper.add_representer(str, _represent_multiline_str)


def dump_paper_yaml(data: dict[str, Any]) -> str:
    """Serialize a paper YAML dict with block-scalar style preserved.

    Intended for *new* paper YAMLs written from scratch (e.g., the GROBID
    bootstrap path), where multi-line strings come from controlled producers
    like _wrap_block_text and have no trailing whitespace, no folded scalars
    to preserve, and no hand-curated list indentation.

    NOT a faithful round-tripper for hand-edited YAMLs: the underlying
    yaml.SafeDumper normalizes folded scalars, list indentation, and blank
    lines, and falls back to double-quoted strings when block scalars
    cannot represent trailing whitespace. For mass edits to existing files,
    use a line-based or AST-based approach.
    """
    return yaml.dump(
        data,
        Dumper=_BlockStyleDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
