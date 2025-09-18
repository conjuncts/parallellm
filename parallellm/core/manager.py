import logging
from typing import List, Optional, Union
from colorama import Fore, Style, init
from parallellm.core.backend import BaseBackend
from parallellm.core.exception import NotAvailable, WrongStage
from parallellm.core.hash import compute_hash
from parallellm.core.response import (
    LLMIdentity,
    LLMDocument,
    LLMResponse,
    ReadyLLMResponse,
)
from parallellm.provider.base import BaseProvider
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger, HashStatus


class StatusDashboard:
    """Context manager for the hash status dashboard"""

    def __init__(self, batch_manager: "BatchManager", log_k: int):
        self._batch_manager = batch_manager
        self._was_displaying = False
        self._log_k = log_k

        # TODO: make log_k work (right now it works on the dash_logger level)

    @property
    def _dash_logger(self):
        return self._batch_manager._dash_logger

    def __enter__(self):
        # Store current display state and enable display
        print()
        self._was_displaying = self._dash_logger.display
        self._dash_logger.set_display(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Print final status of dashboard
        self._dash_logger._update_console()

        # Finalize the line and restore original display state
        self._dash_logger.finalize_line()
        self._dash_logger.set_display(self._was_displaying)
        print()
        # Handle exceptions like ParallellmContext
        if exc_type in (NotAvailable, WrongStage):
            return True
        return False

    def print(self, *args, **kwargs):
        """
        Print to console above the dashboard output.
        This ensures proper display ordering when the dashboard is active.
        """
        # Use the dashboard logger's coordinated print method
        self._dash_logger.coordinated_print(*args, **kwargs)


# New context manager class for BatchManager
class ParalleLLMContext:
    """Context manager for BatchManager lifecycle (default context)"""

    def __init__(self, batch_manager: "BatchManager"):
        self._bm = batch_manager

    def __enter__(self):
        # Any setup logic can go here if needed
        return self._bm

    def __exit__(self, exc_type, exc_value, traceback):
        # Delegate to BatchManager's error handling logic
        if exc_type in (NotAvailable, WrongStage):
            return True
        return False


class BatchManager:
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
        self._current_seq = 0

        # Initialize the hash logger with display disabled by default
        self._dash_logger: DashboardLogger = dash_logger

    def default(self):
        """
        Begins a 'default' stage.
        Usage: `with pllm.default(): ...`
        """
        return ParalleLLMContext(self)

    def dashboard(self, log_k: int = 10):
        """
        Create a context manager for the status dashboard.

        :param log_k: Number of hashes to display in the dashboard (default 10)
        :returns: StatusDashboard context manager with print() method for clean console output
        """
        # TODO implement log_k
        return StatusDashboard(self, log_k=log_k)

    @property
    def current_stage(self):
        return self.metadata["current_stage"]

    @property
    def metadata(self):
        return self._fm.metadata

    @metadata.setter
    def metadata(self, value):
        self._fm.metadata = value

    def when_stage(self, stage_name):
        if stage_name != self.current_stage:
            raise WrongStage()
        self._logger.info(f"Entered stage {Fore.CYAN}{stage_name}{Style.RESET_ALL}")

    def goto_stage(self, stage_name):
        # TODO: TODO: TODO: delay stage change until after the task manager exits.
        self.metadata["current_stage"] = stage_name
        self._logger.info(f"Switched to stage {Fore.CYAN}{stage_name}{Style.RESET_ALL}")

    def save_userdata(self, key, value):
        """
        The intended way to let data persist across stages
        """
        return self._fm.save_userdata(self.current_stage, key, value)

    def load_userdata(self, key):
        """
        The intended way to let data persist across stages
        """
        return self._fm.load_userdata(self.current_stage, key)

    def persist(self):
        """
        Ensure that everything is properly saved
        """
        self._backend.persist()

        self._fm.persist()

    def set_hash_display(self, display: bool):
        """
        Enable or disable hash logger console display

        Args:
            display: Whether to show hash logger output
        """
        self._dash_logger.set_display(display)

    def update_hash_status(self, hash_value: str, status: HashStatus):
        """
        Update the status of a hash in the logger

        Args:
            hash_value: The hash value to update
            status: New status - one of 'C' (cached), '↗' (sent), '↘' (received), '✓' (stored)
        """
        if self._dash_logger is not None:
            self._dash_logger.update_hash(hash_value, status)

    def finalize_hash_display(self):
        """
        Finalize the hash display line and move to next line.
        Useful when you want to ensure subsequent print() calls appear on new lines.
        """
        self._dash_logger.finalize_line()

    def ask_llm(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: Union[LLMIdentity, str, None] = None,
        _hoist_images=None,
        **kwargs,
    ) -> LLMResponse:
        """
        Ask the LLM a question

        :param instructions: The system prompt to use
        :param documents: Documents to use as context. Can be strings or images.
        :param llm: The identity of the LLM to use.
            Can be helpful multi-agent or multi-model scenarios.
        :param _hoist_images: Gemini recommends that images be hoisted to the front of the message.
            Set to True/False to explicitly enforce/disable.
        :returns: A LLMResponse. The value is **lazy loaded**: for best efficiency,
            it should not be resolved until you actually need it.
        """

        if isinstance(llm, str):
            llm = LLMIdentity(llm)

        seq_id = self._current_seq
        self._current_seq += 1

        # Cache using datastore
        hashed = compute_hash(instructions, documents)
        cached = self._backend.retrieve(self.current_stage, hashed, seq_id)
        if cached is not None:
            self.update_hash_status(hashed, HashStatus.CACHED)
            return ReadyLLMResponse(
                stage=self.current_stage,
                seq_id=seq_id,
                doc_hash=instructions,
                value=cached,
            )

        # Not cached, submit to provider
        self.update_hash_status(hashed, HashStatus.SENT)
        return self._provider.submit_query_to_provider(
            instructions,
            documents,
            stage=self.current_stage,
            seq_id=seq_id,
            hashed=hashed,
            llm=llm,
            _hoist_images=_hoist_images,
            **kwargs,
        )

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
