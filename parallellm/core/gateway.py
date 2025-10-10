import logging
from typing import Literal

from parallellm.core.agent.orchestrator import AgentOrchestrator
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger
from parallellm.logging.fancy import parallellm_log_handler


class ParalleLLMGateway:
    def resume_directory(
        self,
        directory,
        *,
        strategy: Literal["sync", "async", "batch", "hybrid"] = "async",
        provider: Literal["openai", None] = None,
        datastore: Literal["sqlite", "sqlite_parquet"] = "sqlite",
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

        # 2. Setup logger
        logger = logging.getLogger("parallellm")
        logger.setLevel(log_level)
        logger.addHandler(parallellm_log_handler)

        # logger.debug("Resuming directory")

        dash_logger = DashboardLogger(k=10, display=False)
        parallellm_log_handler.set_dash_logger(dash_logger)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

        # 3. Setup components
        fm = FileManager(directory)

        # logger.debug("Creating backend")
        if datastore == "sqlite":
            datastore_cls = None  # default
        elif datastore == "sqlite_parquet":
            from parallellm.core.datastore.semi_sql_parquet import (
                SQLiteParquetDatastore,
            )

            datastore_cls = SQLiteParquetDatastore

        if strategy == "async":
            from parallellm.core.backend.async_backend import AsyncBackend

            backend = AsyncBackend(
                fm, dash_logger=dash_logger, datastore_cls=datastore_cls
            )
        elif strategy == "sync":
            from parallellm.core.backend.sync_backend import SyncBackend

            backend = SyncBackend(
                fm, dash_logger=dash_logger, datastore_cls=datastore_cls
            )
        else:
            raise NotImplementedError(f"Strategy '{strategy}' is not implemented yet")

        # logger.debug("Creating provider")
        if provider == "openai":
            from parallellm.provider.openai import (
                AsyncOpenAIProvider,
                SyncOpenAIProvider,
            )

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

        logger.debug("Creating AgentOrchestrator")
        bm = AgentOrchestrator(
            file_manager=fm,
            backend=backend,
            provider=provider,
            logger=logger,
            dash_logger=dash_logger,
        )

        logger.info(f"Resuming with session_id={bm.get_session_counter()}")
        return bm


ParalleLLM = ParalleLLMGateway()
