"""
Athena Tarot — ASCII card art.

Renders ASCII art for drawn cards, shown in the reading. The art is the
MIT-licensed ascii-tarot set by Kathryn Isabelle Lawrence
(https://github.com/lawreka/ascii-tarot), stored in art/rider_waite_ascii.json
with an upright and a reversed rendering for each of the 78 cards. See
art/LICENSE-ascii-tarot.txt for the license and attribution.

The art is keyed by Rider-Waite card names (Page/Knight, "The World"), while the
Golden Dawn deck uses different names for the courts and the last trump
(Princess/Prince, "The Universe"). _art_key() bridges the two so any deck whose
cards resolve to a Rider-Waite name gets art. Art is display-only; it is never
sent to the model.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_ART_PATH = Path(__file__).resolve().parent / "art" / "rider_waite_ascii.json"

# The source art prints the card's name in a cartouche at the bottom of the
# frame: an inner separator, the name line, then the bottom border. On reversed
# cards the name renders as flipped glyphs (e.g. "spɹoʍs ɟo ƃuıʞ"), and only
# named cards (majors, aces, courts) carry it at all — so it reads
# inconsistently. The card is already named in the legend, so we drop it.
_SEP = re.compile(r"^\|-+\|$")       # inner separator:  |-----|
_BOTTOM = re.compile(r"^`-+´$")      # bottom border:    `-----´

# Golden Dawn court ranks -> Rider-Waite court ranks used by the art set.
_RANK_TO_RWS = {
    "Princess": "Page",
    "Prince": "Knight",
    # Queen and King already match.
}

# Golden Dawn card names -> Rider-Waite names, for cards the art keys differently.
_NAME_TO_RWS = {
    "The Universe": "The World",
}

_art_cache: Optional[Dict[str, Dict[str, str]]] = None


def _strip_name_label(art: str) -> str:
    """Remove the card-name cartouche from the bottom of an art frame.

    The cartouche is the last three lines of the frame when they are an inner
    separator, the name line, and the bottom border. Those two upper lines are
    dropped and the bottom border kept, so the frame stays closed. Cards without
    a name cartouche (plain pips) are returned unchanged.
    """
    lines: List[str] = art.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if (
        len(lines) >= 3
        and _BOTTOM.match(lines[-1].strip())
        and _SEP.match(lines[-3].strip())
    ):
        del lines[-3:-1]  # drop the separator and the name line, keep the border
    return "\n".join(lines)


def _load_art() -> Dict[str, Dict[str, str]]:
    """Load and cache the art map: {rws_name: {"upright": str, "reversed": str}}.

    The name cartouche is stripped from every frame on load, so no card shows
    its name inside the art (readable or flipped).
    """
    global _art_cache
    if _art_cache is None:
        with open(_ART_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f).get("art", {})
        _art_cache = {
            name: {
                orient: _strip_name_label(art)
                for orient, art in entry.items()
            }
            for name, entry in raw.items()
        }
    return _art_cache


def _art_key(card: Dict[str, Any]) -> str:
    """Map a deck card to its Rider-Waite art key.

    Handles Golden Dawn court ranks (Princess/Prince) and renamed trumps
    (The Universe). Falls back to the card's own name for anything already in
    Rider-Waite form.
    """
    name = card.get("name", "")
    if name in _NAME_TO_RWS:
        return _NAME_TO_RWS[name]

    rank = card.get("rank")
    suit = card.get("suit")
    if rank and suit:
        rws_rank = _RANK_TO_RWS.get(rank, rank)
        return f"{rws_rank} of {suit}"

    return name


def get_art(card: Dict[str, Any], orientation: str = "upright") -> Optional[str]:
    """Return the ASCII art for a card in the given orientation, or None.

    None is returned if the card cannot be matched to the art set, so callers
    can degrade gracefully (show the reading without art).
    """
    entry = _load_art().get(_art_key(card))
    if not entry:
        return None
    return entry.get(orientation) or entry.get("upright")
