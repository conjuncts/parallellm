import logging
from typing import Literal

from parallellm.core.backend.async_backend import AsyncBackend
from parallellm.core.backend.sync_backend import SyncBackend
from parallellm.core.datastore.sqlite import SQLiteDatastore
from parallellm.core.manager import BatchManager
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger
from parallellm.provider.openai import AsyncOpenAIProvider, SyncOpenAIProvider
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
        dash_logger = DashboardLogger(k=10, display=False)

        if strategy == "async":
            backend = AsyncBackend(fm, dash_logger=dash_logger)
        elif strategy == "sync":
            backend = SyncBackend(fm, dash_logger=dash_logger)
        else:
            raise NotImplementedError(f"Strategy '{strategy}' is not implemented yet")

        if provider == "openai":
            if strategy == "async":
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
                provider = AsyncOpenAIProvider(client=client, backend=backend)
            elif strategy == "sync":
                from openai import OpenAI

                client = OpenAI()
                provider = SyncOpenAIProvider(client=client, backend=backend)
            else:
                # For other strategies, default to async for now
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
                provider = AsyncOpenAIProvider(client=client, backend=backend)

        # Get the parallellm logger and configure it specifically
        logger = logging.getLogger("parallellm")
        logger.setLevel(log_level)
        logger.addHandler(parallellm_log_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

        bm = BatchManager(
            file_manager=fm,
            backend=backend,
            provider=provider,
            logger=logger,
            dash_logger=dash_logger,
        )

        logger.info(
            f"Resuming with session_id={bm.metadata.get('session_counter')}"
            + f" and latest_checkpoint={bm.metadata.get('latest_checkpoint')}"
        )
        return bm


ParalleLLM = ParalleLLMGateway()
