from typing import List, Optional
from parallellm.core.exception import NotAvailable, WrongStage
from parallellm.core.response import LLMIdentity, LLMDocument, LLMResponse
from parallellm.provider.base import BaseProvider
from parallellm.file_io.file_manager import FileManager


class BatchManager:
    def __init__(self, file_manager: FileManager, provider: BaseProvider):
        """
        channel: str, optional
            Parallellm is typically used as a singleton.
            If you want to have multiple instances,
            specify a channel name.
        """
        self._file_manager = file_manager
        self._provider = provider
        self.current_stage = file_manager.current_stage()

    def __enter__(self):
        return self

    def when_stage(self, stage_name):
        if stage_name != self.current_stage:
            raise WrongStage()

    def goto_stage(self, stage_name):
        self.current_stage = stage_name

        # TODO: save to disk

    def save_data(self, key, value):
        """
        The intended way to let data persist across stages
        """
        return self._file_manager.save_data(self.current_stage, key, value)

    def load_data(self, key):
        """
        The intended way to let data persist across stages
        """
        return self._file_manager.load_data(self.current_stage, key)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type in (NotAvailable, WrongStage):
            return True
        return False

    def persist(self):
        """
        Ensure that everything is properly saved
        """
        pass

    def ask_llm(
        self,
        system_prompt,
        documents: List[LLMDocument],
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
        pass