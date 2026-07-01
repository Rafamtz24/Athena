"""
Athena API - Main Entry Point

HTTP entry point for the Athena AI platform.
All business logic flows through AthenaBrain.process().
"""

from fastapi import FastAPI

from athena.brain.brain import AthenaBrain
from athena.pc import get_system_info

# Import settings and logger for convenience at package level
from athena.config.settings import settings
from athena.logging.logger import logger


# Shared AthenaBrain instance for all endpoints
brain = AthenaBrain()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
)


@app.get("/")
async def home():
    """
    Root endpoint - returns application status and metadata.

    Returns:
        dict: Application name, status, and version.
    """
    logger.info("Root endpoint accessed")
    return {
        "name": settings.app_name,
        "status": "running",
        "version": settings.version,
    }


@app.get("/system")
async def system():
    """
    System information endpoint - returns live system metrics.

    Returns:
        dict: Computer name, OS info, CPU stats, RAM usage, disk partitions.
    """
    logger.info("System info endpoint accessed")
    return get_system_info()


@app.get("/process")
async def process(message: str):
    """
    Process a message through the Athena brain pipeline.

    Args:
        message: The input message to process.

    Returns:
        dict: Processing result from the Athena Brain.
    """
    logger.info("Process endpoint accessed with message: %s", message)
    result = await brain.process(message)
    return {"result": result}


@app.get("/debug/last_thought")
async def debug_last_thought():
    """
    Debug endpoint - returns the last processed thought as JSON.

    Returns:
        dict: Thought data or status "empty" if no thought exists yet.
    """
    logger.info("Debug last_thought endpoint accessed")
    thought = brain.debug_manager.get_last_thought()

    if thought is None:
        return {"status": "empty"}

    return {
        "id": thought.id,
        "user_input": thought.user_input,
        "history": thought.history,
        "memories": thought.memories,
        "knowledge": thought.knowledge,
        "plan": thought.plan,
        "response": thought.get_response(),
        "trace": thought.trace,
        "metadata": thought.metadata,
    }