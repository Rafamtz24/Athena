"""
Athena Brain Module

Core orchestration component that receives requests, reasons, delegates to providers,
and returns responses.
"""

from athena.memory.manager import MemoryManager
from athena.thought.models import Thought
from athena.thought.pipeline import ThoughtPipeline


class AthenaBrain:
    """
    The central brain of the Athena AI platform.

    Responsibilities:
        - Receive incoming requests
        - Create Thought objects
        - Pass Thoughts through ThoughtPipeline
        - Return thought.response
        - Manage memory through MemoryManager

    Methods:
        process(message): Process a message and return thought.response.
    """

    def __init__(self) -> None:
        self.memory_manager = MemoryManager()
        self.pipeline = ThoughtPipeline()

    async def process(self, message: str) -> str:
        """
        Process a message through the Athena brain pipeline.

        Creates a Thought object, passes it through ThoughtPipeline,
        and returns the response from the thought.

        Args:
            message: The input message to process.

        Returns:
            The response string from the thought.
        """
        thought = Thought(user_input=message)
        await self.pipeline.process(thought)
        return thought.get_response()


# Singleton instance for convenience
brain = AthenaBrain()