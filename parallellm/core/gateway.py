import logging
from typing import Literal

from parallellm.core.backend.async_backend import AsyncBackend
from parallellm.core.datastore.sqlite import SQLiteDataStore
from parallellm.core.manager import BatchManager
from parallellm.file_io.file_manager import FileManager
from parallellm.provider.openai import AsyncOpenAIProvider
from parallellm.logging.fancy import parallellm_log_handler


class ParalleLLMGateway:
    def resume_directory(
        self,
        directory,
        *,
        strategy: Literal["sync", "async", "batch", "hybrid"] = "async",
        provider: Literal["openai", None] = None,
        dry_run=False,
        log_level=logging.INFO,
    ):
        """
        Resume a BatchManager from a given directory.
        """
        # Logic to resume from the specified directory
        # 1. Validation
        if strategy not in ["sync", "async", "batch", "hybrid"]:
            raise ValueError(f"Unknown strategy '{strategy}'")
        if dry_run:
            raise NotImplementedError("Dry run is not implemented yet")

        # 2. Setup components
        fm = FileManager(directory)

        if strategy == "async":
            backend = AsyncBackend(fm)
        else:
            raise NotImplementedError(f"Strategy '{strategy}' is not implemented yet")
        if provider == "openai":
            if strategy == "async":
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
            else:
                from openai import OpenAI

                client = OpenAI()

            provider = AsyncOpenAIProvider(client=client, backend=backend)

        # Get the parallellm logger and configure it specifically
        logger = logging.getLogger("parallellm")
        logger.setLevel(log_level)
        logger.addHandler(parallellm_log_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False
        return BatchManager(
            file_manager=fm, backend=backend, provider=provider, logger=logger
        )


ParalleLLM = ParalleLLMGateway()
