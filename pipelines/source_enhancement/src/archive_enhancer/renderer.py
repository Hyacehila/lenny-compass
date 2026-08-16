from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TURN_HEADER_RE = re.compile(r"^\*\*(?P<speaker>.+?)\*\*(?: \((?P<timestamp>\d{2}:\d{2}:\d{2})\))?:\s*$")


def render_podcast(path: Path, plan: dict[str, Any]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    starts: dict[int, list[str]] = {}
    for chapter in plan["chapters"]:
        first_section = True
        for section in chapter["sections"]:
            turn_id = int(section["start_turn"])
            additions = starts.setdefault(turn_id, [])
            if first_section:
                additions.append(f"## {chapter['title']}\n")
                first_section = False
            additions.append(f"### {section['title']}\n")

    output: list[str] = []
    turn_id = 0
    for line in lines:
        if TURN_HEADER_RE.match(line.rstrip("\r\n")):
            output.extend(starts.get(turn_id, []))
            turn_id += 1
        output.append(line)
    if turn_id == 0:
        raise ValueError(f"No speaker turns found while rendering {path}")
    path.write_text("".join(output), encoding="utf-8", newline="")
