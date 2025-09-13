import logging
from typing import List, Optional, Union
from colorama import Fore, Style, init
from parallellm.core.exception import NotAvailable, WrongStage
from parallellm.core.response import LLMIdentity, LLMDocument, LLMResponse
from parallellm.provider.base import BaseProvider
from parallellm.file_io.file_manager import FileManager


class BatchManager:
    def __init__(self, file_manager: FileManager, provider: BaseProvider, *, logger):
        """
        channel: str, optional
            Parallellm is typically used as a singleton.
            If you want to have multiple instances,
            specify a channel name.
        """
        self._fm = file_manager
        self._provider = provider
        self._logger = logger

    def __enter__(self):
        return self

    @property
    def metadata(self):
        return self._fm.metadata

    @metadata.setter
    def metadata(self, value):
        self._fm.metadata = value

    def when_stage(self, stage_name):
        if stage_name != self.metadata["current_stage"]:
            raise WrongStage()
        self._logger.info(
            f"{Fore.GREEN}Entered stage '{Fore.CYAN}{stage_name}{Fore.GREEN}'{Style.RESET_ALL}"
        )

    def goto_stage(self, stage_name):
        self.metadata["current_stage"] = stage_name
        self._logger.info(
            f"{Fore.GREEN}Switched to stage '{Fore.CYAN}{stage_name}{Fore.GREEN}'{Style.RESET_ALL}"
        )

        # TODO: save to disk

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
        system_prompt,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: Optional[LLMIdentity] = None,
        _hoist_images=None,
    ) -> LLMResponse:
        """
        Ask the LLM a question

        :param system_prompt: The system prompt to use
        :param documents: Documents to use as context. Can be strings or images.
        :param llm: The identity of the LLM to use.
            Can be helpful multi-agent or multi-model scenarios.
        :param _hoist_images: Gemini recommends that images be hoisted to the front of the message.
            Set to True/False to explicitly enforce/disable.
        :returns: A LLMResponse. The value is **lazy loaded**: for best efficiency,
            it should not be resolved until you actually need it.
        """

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
