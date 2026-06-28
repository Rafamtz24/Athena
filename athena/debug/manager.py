class DebugManager:
    def __init__(self):
        self._last_thought = None

    def set_last_thought(self, thought) -> None:
        self._last_thought = thought

    def get_last_thought(self):
        return self._last_thought
