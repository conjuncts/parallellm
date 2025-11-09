import logging
from typing import Literal

from parallellm.core.agent.orchestrator import AgentOrchestrator
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger
from parallellm.logging.fancy import parallellm_log_handler
from parallellm.provider.openai import BatchOpenAIProvider
from parallellm.types import MinorTweaks, ProviderType


class ParalleLLMGateway:
    def resume_directory(
        self,
        directory,
        *,
        strategy: Literal["sync", "async", "batch", "hybrid"] = "async",
        provider: ProviderType = "openai",
        datastore: Literal["sqlite", "sqlite_parquet"] = "sqlite",
        dry_run=False,
        log_level=logging.INFO,
        ignore_cache=False,
        rewrite_cache=False,
        throttler=None,
        tweaks: MinorTweaks = MinorTweaks(),
    ) -> AgentOrchestrator:
        """
        Resume an AgentOrchestrator from a previously saved directory.

        :param directory: Path to directory

        :param strategy: Execution strategy for LLM calls
        :param provider: LLM provider to use for API calls
        :param datastore: Backend datastore type for response storage. Recommended: sqlite.
        :param dry_run: If True, validate setup without making actual API calls
        :param log_level: Logging level for the session
        :param ignore_cache: If True, always submit to API instead of using cached responses
        :param rewrite_cache: If True, overwrite cached responses with new ones (uses upsert)
        :param throttler: Throttler instance for rate limiting (default: None, no throttling)
        :return: Configured AgentOrchestrator instance
        :raises ValueError: If strategy is not supported
        :raises NotImplementedError: If dry_run is True or strategy is not implemented
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

        logger.debug("Resuming directory")

        dash_logger = DashboardLogger(k=10, display=False)
        parallellm_log_handler.set_dash_logger(dash_logger)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

        # 3. Setup components
        fm = FileManager(directory)

        logger.debug("Creating backend")
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
                fm,
                dash_logger=dash_logger,
                datastore_cls=datastore_cls,
                rewrite_cache=rewrite_cache,
                max_concurrent=tweaks.async_max_concurrent,
                throttler=throttler,
            )
        elif strategy == "sync":
            from parallellm.core.backend.sync_backend import SyncBackend

            backend = SyncBackend(
                fm,
                dash_logger=dash_logger,
                datastore_cls=datastore_cls,
                rewrite_cache=rewrite_cache,
                throttler=throttler,
            )
        elif strategy == "batch":
            from parallellm.core.backend.batch_backend import BatchBackend

            backend = BatchBackend(
                fm,
                dash_logger=dash_logger,
                datastore_cls=datastore_cls,
                session_id=fm._get_session_counter(),
                confirm_batch_submission=tweaks.batch_user_confirmation,
                rewrite_cache=rewrite_cache,
            )
        else:
            raise NotImplementedError(f"Strategy '{strategy}' is not implemented yet")

        logger.debug("Creating provider")
        if provider == "openai":
            from parallellm.provider.openai import (
                AsyncOpenAIProvider,
                SyncOpenAIProvider,
            )

            if strategy == "async":
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
                provider = AsyncOpenAIProvider(client=client)
            elif strategy == "batch":
                from openai import OpenAI

                client = OpenAI()
                provider = BatchOpenAIProvider(client=client)
            else:
                # For other strategies, default to sync for now
                from openai import OpenAI

                client = OpenAI()
                provider = SyncOpenAIProvider(client=client)
        elif provider == "google":
            from parallellm.provider.google import (
                AsyncGoogleProvider,
                BatchGoogleProvider,
                SyncGoogleProvider,
            )
            from google import genai

            if strategy == "async":
                client = genai.Client()
                provider = AsyncGoogleProvider(client=client)
            elif strategy == "batch":
                client = genai.Client()
                provider = BatchGoogleProvider(client=client)
            else:
                client = genai.Client()
                provider = SyncGoogleProvider(client=client)
        elif provider == "anthropic":
            from parallellm.provider.anthropic import (
                AsyncAnthropicProvider,
                SyncAnthropicProvider,
            )

            if strategy == "async":
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic()
                provider = AsyncAnthropicProvider(client=client)
            else:
                from anthropic import Anthropic

                client = Anthropic()
                provider = SyncAnthropicProvider(client=client)
        else:
            raise NotImplementedError(f"Provider '{provider}' not implemented yet")

        logger.debug("Creating AgentOrchestrator")
        bm = AgentOrchestrator(
            file_manager=fm,
            backend=backend,
            provider=provider,
            logger=logger,
            dash_logger=dash_logger,
            ignore_cache=ignore_cache,
            strategy=strategy,
        )

        # try downloading previous batches if any
        if strategy == "batch":
            backend.try_download_all_batches(provider)

        logger.info(f"Resuming with session_id={bm.get_session_counter()}")
        return bm


ParalleLLM = ParalleLLMGateway()
