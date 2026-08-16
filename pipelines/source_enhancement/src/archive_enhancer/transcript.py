from __future__ import annotations

import re
from dataclasses import dataclass


TURN_HEADER_RE = re.compile(r"^\*\*(?P<speaker>.+?)\*\*(?: \((?P<timestamp>\d{2}:\d{2}:\d{2})\))?:\s*$")
GENERATED_HEADING_RE = re.compile(r"^#{1,3}\s+")


@dataclass(frozen=True)
class Turn:
    turn_id: int
    speaker: str
    timestamp: str | None
    text: str
    line_start: int


def parse_turns(text: str) -> list[Turn]:
    lines = text.splitlines()
    starts: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        match = TURN_HEADER_RE.match(line)
        if match:
            starts.append((index, match))
    turns: list[Turn] = []
    for turn_id, (start, match) in enumerate(starts):
        end = starts[turn_id + 1][0] if turn_id + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start + 1:end]).strip()
        turns.append(Turn(turn_id, match.group("speaker"), match.group("timestamp"), body, start))
    if not turns:
        raise ValueError("No speaker turns found")
    return turns


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def numbered_transcript(turns: list[Turn]) -> str:
    blocks: list[str] = []
    for turn in turns:
        timestamp = turn.timestamp or "NO_TIMESTAMP"
        blocks.append(f"TURN {turn.turn_id} | {timestamp} | {turn.speaker}\n{turn.text}")
    return "\n\n".join(blocks)


def remove_generated_headings(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not GENERATED_HEADING_RE.match(line))
