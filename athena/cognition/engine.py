"""Cognitive Engine module.

Provides the CognitiveEngine class, which serves as the entry point
for future cognitive processing strategies and contexts.
"""

import sys
import traceback

from athena.logging.logger import logger
from athena.prompt.builder import PromptBuilder


class CognitiveEngine:
    """Minimal cognitive engine skeleton.

    Currently only provides a pass-through process method.
    Future versions will add reasoning, planning, reflection, etc.
    """

    def __init__(self, provider):
        self.provider = provider

    def process(self, thought):
        """Process the given Thought object.

        Args:
            thought: A Thought instance to be processed.

        Returns:
            The Thought object with cognitive_engine metadata set.
        """
        logger.info("Cognitive Engine Started")
        thought.metadata["cognitive_engine"] = "processed"
        if thought.get_response() is None:
            builder = PromptBuilder()
            reasoning_package = getattr(thought, 'reasoning_package', None)
            if reasoning_package is not None:
                prompt = builder.build(reasoning_package)
            else:
                # Fallback: build from raw thought (backward compatible)
                prompt = builder._build_from_thought(thought)
            thought.trace["prompt"] = {
                "text": prompt
            }
            try:
                response = self.provider.generate(prompt)
                thought.set_response(response)
            except Exception as e:
                # Provider failed — set error trace, provide fallback response
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                print(f"\n[COGNITIVE ENGINE] Provider generate() threw exception:")
                print(f"[COGNITIVE ENGINE] Exception type: {exc_type.__name__}")
                print(f"[COGNITIVE ENGINE] Exception message: {exc_value}")
                print(f"[COGNITIVE ENGINE] Full traceback:\n{tb_str}")
                thought.trace["error"] = {
                    "source": "provider",
                    "message": str(e),
                    "exception_type": exc_type.__name__,
                    "traceback": tb_str,
                }
                thought.set_response("I'm sorry, I'm currently unable to process your request. Please try again later.")
        result = thought
        logger.info("Cognitive Engine Completed")
        return result
