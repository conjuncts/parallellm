from typing import Optional
from parallellm.provider.hardcoded import guess_provider, guess_provider_and_name
from parallellm.types import ProviderType


class LLMIdentity:
    def __init__(
        self,
        identity: str,
        *,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Identify a specific LLM agent.

        :param identity: The identity string of the LLM.
            Can be a canonical name, like "gpt-4o-mini", or a convenient
            nickname like "alex".
        :param provider: The provider of the LLM, if known. For instance, "openai".
        :param model_name: If a nickname is used for identity, the actual model name.
        """
        self.identity = identity

        if provider is None:
            # do some guessing
            provider, model_name = guess_provider_and_name(identity)
        elif model_name is None:
            # if provider is given but not model_name, assume identity is model_name
            model_name = identity
        # else: both provider and model_name are given, use as-is

        self.provider = provider
        self.model_name = model_name
