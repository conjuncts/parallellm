import logging
from parallellm.core.manager import BatchManager
from parallellm.file_io.file_manager import FileManager
from parallellm.provider.openai import SyncOpenAIProvider
from parallellm.logging.fancy import parallellm_log_handler


class ParalleLLMGateway:
    def resume_directory(
        self, directory, *, provider=None, dry_run=False, verbosity="info"
    ):
        """
        Resume a BatchManager from a given directory.
        """
        # Logic to resume from the specified directory
        # TODO
        if provider is None:
            provider = SyncOpenAIProvider()

        # Get the parallellm logger and configure it specifically
        logger = logging.getLogger("parallellm")
        logger.setLevel(logging.INFO)
        logger.addHandler(parallellm_log_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False
        return BatchManager(
            file_manager=FileManager(directory), provider=provider, logger=logger
        )


ParalleLLM = ParalleLLMGateway()
