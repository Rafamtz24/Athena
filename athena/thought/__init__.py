"""
Athena Thought Module

Provides the Thought data model and ThoughtPipeline for processing user requests.
Every future subsystem (memory, reasoning, planning, tools, knowledge, reflection)
operates on this object.
"""

from athena.thought.models import Thought
from athena.thought.pipeline import ThoughtPipeline

__all__ = ["Thought", "ThoughtPipeline"]