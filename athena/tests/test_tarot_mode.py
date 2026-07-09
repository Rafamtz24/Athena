"""
Tests for tarot mode (/tarot).

Tarot mode draws cards at random from a pluggable deck (default: Golden Dawn)
and interprets them for a chosen spread, via a separate path from the Thought
pipeline: no tools, no memory, no extraction. These tests exercise deck loading,
the random draw (count / uniqueness / orientation), prompt assembly, and the
interpretation path using a seeded RNG and a mock provider (no model needed).
"""
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from athena.tarot import reading as tarot
from athena.tarot import art as tarot_art
from athena.prompt.loader import PromptLoader


class MockProvider:
    """Records the last generate() call; no model loaded."""
    def __init__(self):
        self.calls = 0
        self.last_system = None
        self.last_prompt = None

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def get_context_window(self):
        return 4096

    def generate(self, prompt, system=None):
        self.calls += 1
        self.last_prompt = prompt
        self.last_system = system
        return "[mock reading]"


def test_deck_loads_full_78():
    deck = tarot.load_deck("golden_dawn")
    cards = deck["cards"]
    assert len(cards) == 78
    # Names are unique and every card carries upright + reversed meanings.
    names = [c["name"] for c in cards]
    assert len(set(names)) == 78
    assert all(c.get("upright") and c.get("reversed") for c in cards)
    print("  [OK] golden_dawn deck loads 78 unique cards with meanings")


def test_golden_dawn_is_default_and_listed():
    assert "golden_dawn" in tarot.list_decks()
    # Default draw uses the Golden Dawn deck.
    reading = tarot.draw_for_spread(1, rng=random.Random(1))
    assert reading["deck"] == "Golden Dawn"
    print("  [OK] Golden Dawn is available and the default deck")


def test_spreads_have_expected_card_counts():
    expected = {1: 1, 2: 3, 3: 3, 4: 10, 5: 5, 6: 5}
    for spread_id, count in expected.items():
        assert len(tarot.SPREADS[spread_id]["positions"]) == count
    print("  [OK] all six spreads declare the expected number of positions")


def test_draw_returns_one_card_per_position_no_duplicates():
    reading = tarot.draw_for_spread(4, rng=random.Random(42))  # Celtic Cross, 10 cards
    draws = reading["draws"]
    assert len(draws) == 10
    # No card repeats within a single reading (drawn without replacement).
    names = [d["card"]["name"] for d in draws]
    assert len(set(names)) == 10
    # Positions are filled in spread order.
    assert draws[0]["position"] == "Present situation"
    assert draws[-1]["position"] == "Likely outcome"
    print("  [OK] draw fills every position with distinct cards, in order")


def test_orientations_are_valid():
    reading = tarot.draw_for_spread(2, rng=random.Random(7))
    assert all(d["orientation"] in ("upright", "reversed") for d in reading["draws"])
    print("  [OK] every drawn card is either upright or reversed")


def test_reversals_disabled_deals_all_upright():
    reading = tarot.draw_for_spread(4, allow_reversals=False, rng=random.Random(99))
    assert all(d["orientation"] == "upright" for d in reading["draws"])
    print("  [OK] allow_reversals=False deals every card upright")


def test_unknown_spread_raises():
    try:
        tarot.draw_for_spread(9)
    except ValueError:
        print("  [OK] draw_for_spread rejects an unknown spread id")
        return
    raise AssertionError("Expected ValueError for spread id 9")


def test_build_prompt_contains_question_cards_and_meanings():
    reading = tarot.draw_for_spread(1, rng=random.Random(3))
    prompt = tarot.build_tarot_prompt("Will the venture succeed?", reading)
    assert "Question: Will the venture succeed?" in prompt
    card = reading["draws"][0]["card"]
    orientation = reading["draws"][0]["orientation"]
    assert card["name"] in prompt
    assert card[orientation] in prompt  # the orientation-correct meaning is included
    print("  [OK] build_tarot_prompt includes the question, cards and meanings")


def test_build_prompt_general_reading_when_blank():
    reading = tarot.draw_for_spread(1, rng=random.Random(3))
    prompt = tarot.build_tarot_prompt("", reading)
    assert "general reading" in prompt.lower()
    print("  [OK] a blank question yields a general-reading prompt")


def test_interpret_single_call_with_tarot_prompt():
    provider = MockProvider()
    reading = tarot.draw_for_spread(3, rng=random.Random(5))
    answer = tarot.interpret(provider, "What should I focus on?", reading)
    assert answer == "[mock reading]"
    # Exactly one provider call (no tools / no memory / no extra generations).
    assert provider.calls == 1
    assert provider.last_system is not None
    assert "tarot" in provider.last_system.lower()
    print("  [OK] interpret: one grounded call with the tarot system prompt")


def test_format_draws_lists_positions_and_cards():
    reading = tarot.draw_for_spread(2, rng=random.Random(11))
    text = tarot.format_draws(reading)
    assert "Past" in text and "Present" in text and "Future" in text
    assert reading["draws"][0]["card"]["name"] in text
    print("  [OK] format_draws renders positions and drawn cards")


def test_every_deck_card_has_art_both_orientations():
    deck = tarot.load_deck("golden_dawn")
    missing = []
    for card in deck["cards"]:
        for orientation in ("upright", "reversed"):
            if not tarot_art.get_art(card, orientation):
                missing.append((card["name"], orientation))
    assert not missing, f"cards missing art: {missing}"
    print("  [OK] all 78 Golden Dawn cards resolve to art (upright + reversed)")


def test_art_name_mapping_golden_dawn_to_rider_waite():
    # Golden Dawn courts and 'The Universe' must map to Rider-Waite art keys.
    assert tarot_art._art_key({"name": "Princess of Wands", "rank": "Princess", "suit": "Wands"}) == "Page of Wands"
    assert tarot_art._art_key({"name": "Prince of Cups", "rank": "Prince", "suit": "Cups"}) == "Knight of Cups"
    assert tarot_art._art_key({"name": "Queen of Swords", "rank": "Queen", "suit": "Swords"}) == "Queen of Swords"
    assert tarot_art._art_key({"name": "The Universe", "arcana": "major"}) == "The World"
    print("  [OK] Golden Dawn names map onto Rider-Waite art keys")


def test_reversed_art_differs_from_upright():
    card = {"name": "Ten of Swords", "rank": "Ten", "suit": "Swords"}
    up = tarot_art.get_art(card, "upright")
    rev = tarot_art.get_art(card, "reversed")
    assert up and rev and up != rev
    print("  [OK] reversed art is the flipped rendering, not the upright one")


def test_format_draws_embeds_art_and_can_disable():
    reading = tarot.draw_for_spread(1, rng=random.Random(3))
    with_art = tarot.format_draws(reading, include_art=True)
    without = tarot.format_draws(reading, include_art=False)
    # The art frame border appears only when art is included.
    assert ".-----" in with_art
    assert ".-----" not in without
    print("  [OK] format_draws embeds art, and include_art=False omits it")


def test_name_cartouche_stripped_from_all_art():
    import re
    sep = re.compile(r"^\|-+\|$")
    bot = re.compile(r"^`-+´$")

    # A named card must not print its name inside the art (upright or reversed),
    # and the frame must remain closed by the bottom border.
    king = {"name": "King of Swords", "rank": "King", "suit": "Swords"}
    up = tarot_art.get_art(king, "upright")
    rev = tarot_art.get_art(king, "reversed")
    assert "King of Swords" not in up
    assert up.rstrip().endswith("´") and rev.rstrip().endswith("´")

    # No card's art should still end with a separator/name/border cartouche.
    for card in tarot.load_deck("golden_dawn")["cards"]:
        for orientation in ("upright", "reversed"):
            lines = [ln for ln in tarot_art.get_art(card, orientation).split("\n") if ln.strip()]
            has_cartouche = (
                len(lines) >= 3
                and bot.match(lines[-1].strip())
                and sep.match(lines[-3].strip())
            )
            assert not has_cartouche, f"{card['name']} [{orientation}] still labelled"
    print("  [OK] name cartouche stripped from every card's art")


def test_tarot_prompt_profile_loads():
    system_prompt = PromptLoader.get_system_prompt("tarot")
    assert "tarot" in system_prompt.lower()
    assert "already been drawn" in system_prompt.lower()
    print("  [OK] tarot.json prompt profile loads")


def test_tarot_mode_follow_up_loop():
    # Drive the interactive terminal flow: a 3-card reading, a typed one-card
    # follow-up, a blank (general) one-card follow-up, then "tarot" for a fresh
    # reading, then "exit". Assert each step produced the expected draw.
    import builtins
    import contextlib
    import io
    from athena import terminal_chat as tc

    calls = []

    class FakeBrain:
        def tarot_reading(self, question, reading):
            calls.append((question, len(reading["draws"])))
            return "[reading]"

    script = iter([
        "Where next?", "2",   # initial reading: question + 3-card spread
        "This week?",         # typed one-card follow-up
        "",                   # blank -> general one-card follow-up
        "tarot", "New Q", "1",  # new reading: question + one-card spread
        "exit",               # leave tarot mode
    ])
    original_input = builtins.input
    builtins.input = lambda *a, **k: next(script)
    # Capture output in a Unicode buffer so the card art never hits a legacy
    # console code page, and to keep the test's own output clean.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tc._tarot_mode(FakeBrain())
    finally:
        builtins.input = original_input

    assert calls == [
        ("Where next?", 3),  # initial 3-card spread
        ("This week?", 1),   # typed one-card follow-up
        ("", 1),             # blank -> general one-card follow-up
        ("New Q", 1),        # "tarot" -> new one-card reading
    ], f"unexpected reading sequence: {calls}"
    print("  [OK] tarot mode: follow-up questions, blank readings, 'tarot', and 'exit'")


def test_draw_uses_system_random_by_default():
    # Without a seeded rng the draw must still succeed and be non-degenerate.
    reading = tarot.draw_for_spread(2)
    assert len(reading["draws"]) == 3
    print("  [OK] default draw uses the system RNG (OS entropy)")


if __name__ == "__main__":
    tests = [
        test_deck_loads_full_78,
        test_golden_dawn_is_default_and_listed,
        test_spreads_have_expected_card_counts,
        test_draw_returns_one_card_per_position_no_duplicates,
        test_orientations_are_valid,
        test_reversals_disabled_deals_all_upright,
        test_unknown_spread_raises,
        test_build_prompt_contains_question_cards_and_meanings,
        test_build_prompt_general_reading_when_blank,
        test_interpret_single_call_with_tarot_prompt,
        test_format_draws_lists_positions_and_cards,
        test_every_deck_card_has_art_both_orientations,
        test_art_name_mapping_golden_dawn_to_rider_waite,
        test_reversed_art_differs_from_upright,
        test_format_draws_embeds_art_and_can_disable,
        test_name_cartouche_stripped_from_all_art,
        test_tarot_prompt_profile_loads,
        test_tarot_mode_follow_up_loop,
        test_draw_uses_system_random_by_default,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\nResults: {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
