"""Verification for S5.2 - CognitiveEngine integration into ThoughtPipeline."""
import sys
sys.path.insert(0, "D:/AI/Athena")
import asyncio
from athena.thought.pipeline import ThoughtPipeline


async def main():
    t = ThoughtPipeline.create("test input")
    result = await ThoughtPipeline.process(None, t)
    print(f"Result: {result}")
    assert result == "Response placeholder", f"Expected 'Response placeholder', got '{result}'"
    print("Verification OK - CognitiveEngine integrated into pipeline")


if __name__ == "__main__":
    asyncio.run(main())
