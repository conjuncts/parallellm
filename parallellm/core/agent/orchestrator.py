from typing import List, Optional, Union
from parallellm.core.agent.agent import AgentContext, AgentDashboardContext
from parallellm.core.backend import BaseBackend
from parallellm.core.response import (
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.provider.base import BaseProvider
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger


class AgentOrchestrator:
    def __init__(
        self,
        file_manager: FileManager,
        backend: BaseBackend,
        provider: BaseProvider,
        *,
        logger,
        dash_logger,
    ):
        """
        channel: str, optional
            Parallellm is typically used as a singleton.
            If you want to have multiple instances,
            specify a channel name.
        dash_logger_k: int, optional
            Number of hashes to display in the hash logger (default 10)
        """
        self._backend = backend
        self._fm = file_manager
        self._provider = provider
        self._logger = logger

        # Initialize the hash logger with display disabled by default
        self._dash_logger: DashboardLogger = dash_logger

    def agent(self, name: str = None, *, dashboard=False):
        if dashboard:
            return AgentDashboardContext(name, self, log_k=10)
        return AgentContext(name, self)

    def save_userdata(self, key, value):
        """
        The intended way to let data persist across checkpoints
        """
        return self._fm.save_userdata(key, value)

    def load_userdata(self, key):
        """
        The intended way to let data persist across checkpoints
        """
        data = self._fm.load_userdata(key)

        # If the loaded data is an LLMResponse, inject the backend
        if isinstance(data, PendingLLMResponse):
            data._backend = self._backend
        elif isinstance(data, ReadyLLMResponse):
            data.value = self._backend.retrieve(data.call_id)

        return data

    def persist(self):
        """
        Ensure that everything is properly saved
        """
        self._backend.persist()

        self._fm.persist()

    def save_to_file(
        self,
        responses: List[LLMResponse],
        fname: str,
        *,
        format: str = "batch-openai",
    ):
        """
        Save a list of responses to a file.
        Creates an openai batch file.
        """
        raise NotImplementedError

    def get_session_counter(self):
        """
        Get session counter, aka session ID."""
        return self._fm.metadata["session_counter"]
