from logging import Logger
from typing import List, Literal, Optional, Union
from parallellm.core.agent.agent import AgentContext, AgentDashboardContext
from parallellm.core.backend import BaseBackend
from parallellm.core.exception import IntegrityError
from parallellm.core.response import (
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.provider.base import BaseProvider
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger
from parallellm.types import AskParameters


class AgentOrchestrator:
    def __init__(
        self,
        file_manager: FileManager,
        backend: BaseBackend,
        provider: BaseProvider,
        *,
        logger: Logger,
        dash_logger,
        ask_params: Optional[AskParameters] = None,
        ignore_cache: bool = False,
        strategy: Optional[Literal["sync", "async", "batch"]] = None,
    ):
        """
        Initialize the AgentOrchestrator.

        :param file_manager: File manager for handling persistence and metadata
        :param backend: Backend for data storage and retrieval
        :param provider: Provider for submitting queries to LLM APIs
        :param logger: Logger instance
        :param dash_logger: Dashboard logger for pretty printing hash status
        :param ask_params: Default parameters for ask_llm() calls
        :param ignore_cache: If True, always submit to the API instead of using cached responses
        """
        self._backend = backend
        self._fm = file_manager
        self._provider = provider
        self._logger = logger

        # Initialize the hash logger with display disabled by default
        self._dash_logger: DashboardLogger = dash_logger

        self.ask_params = ask_params or {}
        self.ignore_cache = ignore_cache
        self.strategy = strategy

    def __enter__(self):
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager, automatically calling persist()."""
        self.persist()
        return False

    def agent(
        self,
        name: str = "default-agent",
        *,
        dashboard=False,
        ask_params: Optional[AskParameters] = None,
    ):
        if ask_params is None:
            ask_params = self.ask_params

        if dashboard:
            return AgentDashboardContext(
                name,
                self,
                log_k=10,
                ask_params=ask_params,
                ignore_cache=self.ignore_cache,
            )
        return AgentContext(
            name, self, ask_params=ask_params, ignore_cache=self.ignore_cache
        )

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
            parsed_response = self._backend.retrieve(data.call_id)
            if parsed_response is None:
                raise IntegrityError("Cached value is no longer available")
            data.value = parsed_response.text

        return data

    def persist(self):
        """
        Ensure that everything is properly saved AND cleans up resources.
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
