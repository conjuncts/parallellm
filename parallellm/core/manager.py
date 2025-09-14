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


class BatchManager:
    def __init__(
        self,
        file_manager: FileManager,
        backend: BaseBackend,
        provider: BaseProvider,
        *,
        logger,
    ):
        """
        channel: str, optional
            Parallellm is typically used as a singleton.
            If you want to have multiple instances,
            specify a channel name.
        """
        self._backend = backend
        self._fm = file_manager
        self._provider = provider
        self._logger = logger
        self._current_seq = 0

    def __enter__(self):
        return self

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
        self._logger.info(
            f"{Fore.GREEN}Entered stage '{Fore.CYAN}{stage_name}{Fore.GREEN}'{Style.RESET_ALL}"
        )

    def goto_stage(self, stage_name):
        # TODO: TODO: TODO: delay stage change until after the task manager exits.
        self.metadata["current_stage"] = stage_name
        self._logger.info(
            f"{Fore.GREEN}Switched to stage '{Fore.CYAN}{stage_name}{Fore.GREEN}'{Style.RESET_ALL}"
        )

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

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type in (NotAvailable, WrongStage):
            return True
        return False

    def persist(self):
        """
        Ensure that everything is properly saved
        """
        self._fm.persist()

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
            self._logger.info(f"{Fore.YELLOW}C {hashed[:8]}{Style.RESET_ALL}")
            return ReadyLLMResponse(
                stage=self.current_stage,
                seq_id=seq_id,
                doc_hash=instructions,
                value=cached,
            )

        # Not cached, submit to provider
        self._logger.info(f"{Fore.YELLOW}↗ {hashed[:8]}{Style.RESET_ALL}")
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
