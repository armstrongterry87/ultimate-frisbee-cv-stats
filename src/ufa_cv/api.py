"""Client for the UFA public stats API.

The API is the supervision signal for this project: it publishes throw-level events with
field coordinates for every game, which is what we train and validate against.

Endpoints reject missing parameters with HTTP 400 and a JSON body naming the parameter,
so the surface is self-documenting.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

BASE_URL = "https://www.backend.ufastats.com/api/v1/"
_USER_AGENT = "ufa-cv/0.1 (ultimate-frisbee-cv-stats)"


class EventType(IntEnum):
    """Event ``type`` codes in ``gameEvents``.

    Codes marked verified were confirmed against a known final score on game
    ``2026-08-08-NY-BOS``: per-side ``GOAL_THROWN`` counts reproduced 24-21 exactly, and
    ``OPPONENT_SCORED`` mirrored the opposing side's total.
    """

    # Line set for a point; carries ``line`` = the seven players on the field.
    LINE_OFFENSE = 1
    LINE_DEFENSE = 2
    LINE_3 = 3
    LINE_5 = 5
    LINE_25 = 25

    PULL = 7               # pullX/pullY/pullMs/puller
    PULL_NO_COORDS = 8     # puller only (out of bounds)

    BLOCK = 11             # defender

    # No payload beyond type/time/timestamp; meaning not established. Do not rely on these.
    UNKNOWN_13 = 13
    UNKNOWN_16 = 16
    UNKNOWN_17 = 17

    OPPONENT_SCORED = 15   # verified: mirrors the other side's goal total
    PASS = 18              # verified: thrower/receiver + coordinates
    GOAL_THROWN = 19       # verified: per-side count equals that side's goals
    DROP = 20              # thrower/receiver + coordinates; inferred, not verified
    THROWAWAY = 22         # thrower + turnoverX/turnoverY

    END_Q1 = 28
    END_Q2 = 29
    END_Q3 = 30
    END_Q4 = 31


#: Events that represent a throw and carry both endpoints in field coordinates.
THROW_TYPES = frozenset(
    {EventType.PASS, EventType.GOAL_THROWN, EventType.DROP, EventType.THROWAWAY}
)

#: Events that set a line, carrying the seven players on the field for the point.
LINE_TYPES = frozenset(
    {
        EventType.LINE_OFFENSE,
        EventType.LINE_DEFENSE,
        EventType.LINE_3,
        EventType.LINE_5,
        EventType.LINE_25,
    }
)

#: Quarter-end markers, in order.
QUARTER_END_TYPES = (
    EventType.END_Q1,
    EventType.END_Q2,
    EventType.END_Q3,
    EventType.END_Q4,
)


class UFAAPIError(RuntimeError):
    pass


def _get(endpoint: str, **params: str) -> Any:
    url = BASE_URL + endpoint
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:  # the body names the offending parameter
        raise UFAAPIError(f"{endpoint} -> HTTP {exc.code}: {exc.read().decode()}") from exc
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def fetch_games(*, date: str | None = None, game_ids: str | None = None) -> list[dict]:
    """Games for a ``YYYY-MM-DD`` date, or specific comma-separated game IDs."""
    if not (date or game_ids):
        raise ValueError("one of date or game_ids is required")
    return _get("games", **({"date": date} if date else {"gameIDs": game_ids}))


def fetch_game_events(game_id: str) -> dict[str, list[dict]]:
    """Throw-level events for one game, as ``{"homeEvents": [...], "awayEvents": [...]}``.

    Note the parameter is ``gameID`` singular; ``gameIDs`` is rejected by this endpoint.
    """
    return _get("gameEvents", gameID=game_id)


@dataclass(frozen=True)
class Goal:
    """A goal, as the pairing of a scoring throw with the side that threw it."""

    timestamp: int
    side: str          # "home" or "away"
    thrower: str
    receiver: str
    receiver_x: float
    receiver_y: float


def extract_goals(events: dict[str, list[dict]]) -> list[Goal]:
    """Every goal in the game, ordered by wall-clock timestamp.

    Reads only ``GOAL_THROWN``; ``OPPONENT_SCORED`` is the same goal mirrored onto the
    other side's stream, so counting both would double every score.
    """
    goals = [
        Goal(
            timestamp=event["timestamp"],
            side=side,
            thrower=event["thrower"],
            receiver=event["receiver"],
            receiver_x=event["receiverX"],
            receiver_y=event["receiverY"],
        )
        for side in ("home", "away")
        for event in events[f"{side}Events"]
        if event["type"] == EventType.GOAL_THROWN and "timestamp" in event
    ]
    goals.sort(key=lambda goal: goal.timestamp)
    return goals


def quarter_end_timestamps(events: dict[str, list[dict]]) -> list[int]:
    """Wall-clock timestamps of the four quarter ends, in order."""
    marks: dict[int, int] = {}
    for side in ("homeEvents", "awayEvents"):
        for event in events[side]:
            if event["type"] in QUARTER_END_TYPES and "timestamp" in event:
                marks[event["type"]] = event["timestamp"]
    return [marks[t] for t in QUARTER_END_TYPES if t in marks]
