"""
Athena Tarot Reading

Tarot mode: draw cards at random from a pluggable deck and interpret them for a
chosen spread. The default deck is the Golden Dawn (Book T) tarot; more decks
can be added by dropping a JSON file in the ``decks/`` directory.

Two guarantees shape this module:

1. The cards are drawn by a system random number generator (``secrets`` /
   ``random.SystemRandom``, seeded from OS entropy), NOT by the language model.
   The draw happens BEFORE any interpretation, so the model can never handpick
   or nudge which cards appear — it only reads what chance dealt.

2. This path is deliberately separate from the normal Thought pipeline: no
   tools, no web search, no memory retrieval, no knowledge extraction. The
   model concentrates solely on interpreting the drawn cards.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Decks ─────────────────────────────────────────────────────────

_DECKS_DIR = Path(__file__).resolve().parent / "decks"
_DEFAULT_DECK = "golden_dawn"


def _deck_path(name: str) -> Path:
    return _DECKS_DIR / f"{name}.json"


def list_decks() -> List[str]:
    """Return the deck names available in the decks directory, sorted."""
    if not _DECKS_DIR.exists():
        return []
    return sorted(p.stem for p in _DECKS_DIR.glob("*.json"))


def load_deck(name: str = _DEFAULT_DECK) -> Dict[str, Any]:
    """Load a deck definition by name.

    Returns the parsed deck dict, which contains a ``name`` and a ``cards``
    list. Raises FileNotFoundError if the deck does not exist, or ValueError if
    the deck is malformed / does not contain the full 78 cards.
    """
    path = _deck_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Deck '{name}' not found at {path}. "
            f"Available decks: {', '.join(list_decks()) or '(none)'}."
        )
    with open(path, "r", encoding="utf-8") as f:
        deck = json.load(f)

    cards = deck.get("cards")
    if not isinstance(cards, list) or not cards:
        raise ValueError(f"Deck '{name}' contains no cards.")
    if len(cards) != 78:
        raise ValueError(
            f"Deck '{name}' has {len(cards)} cards; a tarot deck must have 78."
        )
    return deck


# ── Spreads ───────────────────────────────────────────────────────
# Each spread lists its positions in draw order. The number of positions is the
# number of cards drawn. ``meaning`` explains what the position represents.

SPREADS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "One-Card Pull",
        "positions": [
            {"name": "Guidance", "meaning": "Guidance / answer / the energy of the situation"},
        ],
    },
    2: {
        "name": "Past - Present - Future",
        "positions": [
            {"name": "Past", "meaning": "Past influences"},
            {"name": "Present", "meaning": "Present situation"},
            {"name": "Future", "meaning": "Likely future if the current course continues"},
        ],
    },
    3: {
        "name": "Mirror - Medicine - Message",
        "positions": [
            {"name": "Mirror", "meaning": "Your current state"},
            {"name": "Medicine", "meaning": "What you need to heal, develop, or integrate"},
            {"name": "Message", "meaning": "Guidance from your higher self or the Tarot"},
        ],
    },
    4: {
        "name": "Celtic Cross",
        "positions": [
            {"name": "Present situation", "meaning": "The heart of the matter now"},
            {"name": "Challenge / Obstacle", "meaning": "What crosses or challenges you"},
            {"name": "Foundation / Root cause", "meaning": "The basis or root of the situation"},
            {"name": "Recent past", "meaning": "What is passing away or recently behind you"},
            {"name": "Conscious goal", "meaning": "Your aim, or what is coming into awareness"},
            {"name": "Near future", "meaning": "What is approaching next"},
            {"name": "Yourself", "meaning": "Your attitude and role in the matter"},
            {"name": "Environment / External influences", "meaning": "The people and forces around you"},
            {"name": "Hopes and fears", "meaning": "Your inner hopes and fears about the outcome"},
            {"name": "Likely outcome", "meaning": "Where the current course leads"},
        ],
    },
    5: {
        "name": "Decision-Making Spread",
        "positions": [
            {"name": "Surface of Option A", "meaning": "The apparent nature of Option A"},
            {"name": "True outcome of Option A", "meaning": "The deeper reality / real outcome of Option A"},
            {"name": "Surface of Option B", "meaning": "The apparent nature of Option B"},
            {"name": "True outcome of Option B", "meaning": "The deeper reality / real outcome of Option B"},
            {"name": "What you need to know", "meaning": "What you must understand before deciding"},
        ],
    },
    6: {
        "name": "Relationship Spread",
        "positions": [
            {"name": "Them", "meaning": "The other person"},
            {"name": "You", "meaning": "You in the relationship"},
            {"name": "The relationship", "meaning": "Current state of the relationship"},
            {"name": "Past influences", "meaning": "Past influences shaping it"},
            {"name": "Future potential", "meaning": "Where the relationship may be heading"},
        ],
    },
}


# The reference menu shown when entering tarot mode. Kept as a literal so the
# terminal can print it verbatim.
SPREADS_REFERENCE = """\
============================================================
TAROT SPREADS REFERENCE

1. ONE-CARD PULL
   Best for: daily guidance, clarifying a question, quick readings.
   [1]  ->  Guidance / Answer / Energy of the situation

2. PAST - PRESENT - FUTURE (3 cards)
   Best for: understanding a situation, seeing how events connect.
   [1] Past   [2] Present   [3] Future

3. MIRROR - MEDICINE - MESSAGE (3 cards)
   Best for: self-reflection, personal growth, shadow work.
   [1] Mirror (your current state)
   [2] Medicine (what you need to heal or integrate)
   [3] Message (guidance from your higher self / the Tarot)

4. CELTIC CROSS (10 cards)
   Best for: complex situations, major decisions, deep insight.
   1 Present  2 Challenge  3 Foundation  4 Recent past  5 Conscious goal
   6 Near future  7 Yourself  8 Environment  9 Hopes & fears  10 Outcome

5. DECISION-MAKING SPREAD (5 cards)
   Best for: choosing between two options.
   1 Surface of A   2 True outcome of A
   3 Surface of B   4 True outcome of B
   5 What you need to know before deciding

6. RELATIONSHIP SPREAD (5 cards)
   Best for: romance, friendship, family, work relationships.
   1 Them  2 You  3 The relationship  4 Past influences  5 Future potential
============================================================"""


def get_spread(spread_id: int) -> Optional[Dict[str, Any]]:
    """Return the spread definition for an id (1-6), or None if unknown."""
    return SPREADS.get(spread_id)


# ── Drawing ───────────────────────────────────────────────────────

def draw_for_spread(
    spread_id: int,
    deck_name: str = _DEFAULT_DECK,
    allow_reversals: bool = True,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Draw the cards for a spread.

    The cards are chosen with a system random generator seeded from OS entropy
    (``random.SystemRandom``), so the language model has no influence over which
    cards appear or their orientation. Cards are drawn WITHOUT replacement
    (no card repeats within a reading). Each drawn card is randomly upright or
    reversed unless ``allow_reversals`` is False, in which case all cards are
    dealt upright.

    Args:
        spread_id: 1-6, selecting a spread from SPREADS.
        deck_name: The deck to draw from (defaults to the Golden Dawn deck).
        allow_reversals: When False, every card is dealt upright (no reversals).
        rng: Optional random source, for deterministic tests. Production callers
             leave this None so a fresh SystemRandom (OS entropy) is used.

    Returns:
        A dict with ``spread`` (the spread definition), ``deck`` (deck display
        name) and ``draws``: a list of {position, meaning, card, orientation},
        one per position, in order.
    """
    spread = get_spread(spread_id)
    if spread is None:
        raise ValueError(f"Unknown spread id: {spread_id!r} (choose 1-6).")

    deck = load_deck(deck_name)
    cards = deck["cards"]

    # SystemRandom draws from OS entropy — not reproducible, not model-driven.
    source = rng if rng is not None else random.SystemRandom()

    positions = spread["positions"]
    indices = source.sample(range(len(cards)), len(positions))

    draws: List[Dict[str, Any]] = []
    for position, card_index in zip(positions, indices):
        reversed_ = bool(source.getrandbits(1)) if allow_reversals else False
        draws.append({
            "position": position["name"],
            "meaning": position["meaning"],
            "card": cards[card_index],
            "orientation": "reversed" if reversed_ else "upright",
        })

    return {
        "spread": spread,
        "deck": deck.get("name", deck_name),
        "draws": draws,
    }


def _legend(reading: Dict[str, Any]) -> List[str]:
    """Numbered list mapping each position to the card that fell there."""
    out = []
    for index, draw in enumerate(reading["draws"], start=1):
        orient = "" if draw["orientation"] == "upright" else " (reversed)"
        out.append(f"  {index}. {draw['position']}: {draw['card']['name']}{orient}")
    return out


def format_draws(
    reading: Dict[str, Any],
    include_art: bool = True,
    width: Optional[int] = None,
) -> str:
    """Render the drawn cards as a plain-text layout the user can read directly.

    This is shown BEFORE the interpretation so the user sees exactly which cards
    chance dealt, independent of anything the model then says about them.

    When ``include_art`` is True the cards are laid out as ASCII-art columns,
    packed side by side as many per row as the terminal width allows, each
    captioned ``[n]`` and followed by a numbered legend of positions. When
    False, a simple one-line-per-card list is returned. ``width`` overrides the
    detected terminal width (mainly for tests).
    """
    header = [f"{reading['deck']} - {reading['spread']['name']}", ""]

    if not include_art:
        return "\n".join(header + _legend(reading)).rstrip() + "\n"

    from athena.tarot.art import get_art

    gap = 3
    draws = reading["draws"]

    # Build each card's art as a list of lines; fall back to a labelled box.
    columns: List[List[str]] = []
    for draw in draws:
        art = get_art(draw["card"], draw["orientation"])
        if art:
            columns.append(art.split("\n"))
        else:
            orient = "" if draw["orientation"] == "upright" else " (rev)"
            columns.append([f"{draw['card']['name']}{orient}"])

    # A single column width for every card keeps the grid aligned.
    card_width = max((len(ln) for col in columns for ln in col), default=0)
    card_width = max(card_width, len("[00]"))

    if width is None:
        import shutil
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
    per_row = max(1, (width + gap) // (card_width + gap))

    def pad(col: List[str], height: int) -> List[str]:
        padded = [ln.ljust(card_width) for ln in col]
        padded += [" " * card_width] * (height - len(padded))
        return padded

    body: List[str] = []
    for start in range(0, len(columns), per_row):
        group = columns[start:start + per_row]
        # Caption row: [1] [2] [3] ... centered over each card.
        captions = [f"[{start + offset + 1}]".center(card_width)
                    for offset in range(len(group))]
        body.append((" " * gap).join(captions).rstrip())
        # Art rows: pad every card in the group to the tallest one, then stitch.
        height = max(len(col) for col in group)
        padded = [pad(col, height) for col in group]
        for row in range(height):
            body.append((" " * gap).join(col[row] for col in padded).rstrip())
        body.append("")

    lines = header + body + _legend(reading)
    return "\n".join(lines).rstrip() + "\n"


# ── Prompt assembly & interpretation ──────────────────────────────

def build_tarot_prompt(question: str, reading: Dict[str, Any]) -> str:
    """Assemble the tarot-mode user prompt from the question and the draw.

    The prompt contains only what the model may work with: the question (or a
    note that none was asked), and, for each position, the card that was drawn,
    its orientation, its Golden Dawn title, and the divinatory meaning for that
    orientation. The model must not change which cards appeared.
    """
    if question.strip():
        header = f"Question: {question.strip()}"
    else:
        header = "No question was asked — give a general reading."

    blocks = []
    for index, draw in enumerate(reading["draws"], start=1):
        card = draw["card"]
        orientation = draw["orientation"]
        meaning = card.get(orientation, "")
        title = card.get("title", "")
        blocks.append(
            f"Position {index} - {draw['position']} "
            f"({draw['meaning']}):\n"
            f"  Card: {card['name']} ({orientation})\n"
            f"  Title: {title}\n"
            f"  Meaning ({orientation}): {meaning}"
        )

    cards_block = "\n\n".join(blocks)
    return (
        f"Spread: {reading['spread']['name']} "
        f"(deck: {reading['deck']})\n"
        f"{header}\n\n"
        "The following cards have already been drawn at random for this "
        "reading. Interpret them exactly as dealt.\n\n"
        f"{cards_block}"
    )


def interpret(provider, question: str, reading: Dict[str, Any]) -> str:
    """Interpret an already-drawn spread, grounded strictly in the cards.

    Uses the provider directly (no Thought pipeline, no tools, no memory). The
    cards are NOT drawn here — they were drawn beforehand by draw_for_spread and
    passed in — so the model reads only what chance produced.
    """
    from athena.prompt.loader import PromptLoader

    if not reading.get("draws"):
        return "No cards were drawn, so there is nothing to read."

    system_prompt = PromptLoader.get_system_prompt("tarot")
    prompt = build_tarot_prompt(question, reading)
    return provider.generate(prompt, system=system_prompt)
