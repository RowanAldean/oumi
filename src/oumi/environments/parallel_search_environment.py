# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment that executes web searches with the Parallel Search API."""

from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic import JsonValue

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError
from oumi.core.registry import register_environment
from oumi.environments.endpoint_environment import (
    EndpointEnvironment,
    EndpointStatusError,
    JsonHttpClient,
    JsonHttpProtocol,
    RemoteToolCall,
    RequestsJsonClient,
)
from oumi.environments.utils import parse_env_kwargs

_DEFAULT_ENDPOINT_URL = "https://api.parallel.ai/v1/search"
_DEFAULT_API_KEY_ENV_VARNAME = "PARALLEL_API_KEY"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RETRIES = 3


@dataclass
class ParallelSearchEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for :class:`ParallelSearchEnvironment`."""

    api_key: str | None = None
    """Parallel API key, or ``None`` to read ``api_key_env_varname``."""

    api_key_env_varname: str = _DEFAULT_API_KEY_ENV_VARNAME
    """Environment variable containing the Parallel API key."""

    endpoint_url: str = _DEFAULT_ENDPOINT_URL
    """Parallel Search API endpoint."""

    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    """How long to wait for one search response."""

    max_retries: int = _DEFAULT_MAX_RETRIES
    """Retries for a connection failure or retryable status."""

    def __finalize_and_validate__(self) -> None:
        """Validates the Parallel Search endpoint configuration."""
        if not self.api_key_env_varname:
            raise ValueError("api_key_env_varname is required.")
        if not self.endpoint_url:
            raise ValueError("endpoint_url is required.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

    def resolve_api_key(self) -> str:
        """Returns the configured API key or reads it from the environment."""
        api_key = self.api_key or os.environ.get(self.api_key_env_varname)
        if not api_key:
            raise ValueError(
                "Parallel Search API key is required. Set "
                f"{self.api_key_env_varname} or configure env_kwargs.api_key."
            )
        return api_key


class ParallelSearchProtocol:
    """Sends tool arguments as a Parallel Search API request."""

    def __init__(self, http_client: JsonHttpClient) -> None:
        """Sends requests over the authenticated Parallel client."""
        self._http_client = http_client

    def call(self, request: RemoteToolCall) -> JsonValue:
        """Posts search arguments and returns the Parallel response."""
        try:
            return self._http_client.post_json(request.arguments)
        except EndpointStatusError as error:
            if JsonHttpProtocol._is_tool_refusal(error.status_code):
                raise ToolError(str(error)) from error
            raise


@register_environment("parallel_search")
class ParallelSearchEnvironment(EndpointEnvironment):
    """Environment that executes tool calls with Parallel Search."""

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> ParallelSearchEnvironment:
        """Builds an authenticated Parallel Search environment."""
        kwargs = parse_env_kwargs(
            ParallelSearchEnvironmentKwargs,
            params,
            env_label="ParallelSearchEnvironment",
        )
        client = RequestsJsonClient(
            kwargs.endpoint_url,
            kwargs.timeout_seconds,
            kwargs.max_retries,
            headers={
                "Content-Type": "application/json",
                "x-api-key": kwargs.resolve_api_key(),
            },
        )
        return cls(params, ParallelSearchProtocol(client))
