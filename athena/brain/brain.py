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
        self.pipeline = ThoughtPipeline(self.memory_manager)
        self.history: list[str] = []  # Conversation history stored by the brain

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
        # Copy current conversation history into the thought before processing
        thought.history = list(self.history)

        await self.pipeline.process(thought)
        response = thought.get_response()

        # Append conversation turn to history after obtaining response
        self.history.append(f"User: {message}")
        self.history.append(f"Assistant: {response}")

        # Store in episodic memory
        content = f"User:\n{message}\n\nAssistant:\n{response}"
        self.memory_manager.remember(content)

        return response


# Singleton instance for convenience
brain = AthenaBrain()