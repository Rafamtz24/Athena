"""
Diagnostic script — traces the exact exception in the reproduction scenario.

Reproduction:
    1. "hello"                     → works
    2. "my name is actually rafael" → permanently breaks Athena

This script runs both commands with full pipeline instrumentation
and prints the exact exception and stack trace at every stage.
"""

import asyncio
import sys
import traceback

# ── Force unbuffered stdout so we see all prints in real time ──
sys.stdout.reconfigure(line_buffering=True)

from athena.brain.brain import AthenaBrain


async def reproduce():
    brain = AthenaBrain()

    print("=" * 70)
    print("REPRODUCTION: 'hello'")
    print("=" * 70)

    try:
        r1 = await brain.process("hello")
        print(f"\n>>> Response: {r1!r}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"\n>>> CRASHED on 'hello'!")
        print(f">>> Exception type: {exc_type.__name__}")
        print(f">>> Exception: {exc_value}")
        print(f">>> Traceback:\n{tb_str}")

    print()
    print("=" * 70)
    print("REPRODUCTION: 'my name is actually rafael'")
    print("=" * 70)

    try:
        r2 = await brain.process("my name is actually rafael")
        print(f"\n>>> Response: {r2!r}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"\n>>> CRASHED on 'my name is actually rafael'!")
        print(f">>> Exception type: {exc_type.__name__}")
        print(f">>> Exception: {exc_value}")
        print(f">>> Traceback:\n{tb_str}")

    print()
    print("=" * 70)
    print("POST-CRASH VERIFICATION: 'hello'")
    print("=" * 70)

    try:
        r3 = await brain.process("hello")
        print(f"\n>>> Response: {r3!r}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"\n>>> CRASHED on 'hello' (POST-CRASH)!")
        print(f">>> Exception type: {exc_type.__name__}")
        print(f">>> Exception: {exc_value}")
        print(f">>> Traceback:\n{tb_str}")

    print()
    print("=" * 70)
    print("POST-CRASH VERIFICATION: /system do a health check")
    print("=" * 70)

    try:
        r4 = await brain.process("/system do a health check")
        print(f"\n>>> Response: {r4!r}")
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"\n>>> CRASHED on '/system' (POST-CRASH)!")
        print(f">>> Exception type: {exc_type.__name__}")
        print(f">>> Exception: {exc_value}")
        print(f">>> Traceback:\n{tb_str}")

    print()
    print("=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(reproduce())
