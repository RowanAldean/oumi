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

import pytest
import requests
from pydantic import JsonValue

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError, ToolParams
from oumi.core.registry import REGISTRY, RegistryType
from oumi.environments.endpoint_environment import (
    EndpointCallError,
    EndpointStatusError,
)
from oumi.environments.parallel_search_environment import (
    ParallelSearchEnvironment,
    ParallelSearchEnvironmentKwargs,
    ParallelSearchProtocol,
)

_SEARCH_URL = "https://api.parallel.ai/v1/search"
_RESPONSE = {
    "search_id": "search_123",
    "results": [
        {
            "url": "https://example.com",
            "title": "Example",
            "publish_date": None,
            "excerpts": ["Relevant excerpt."],
        }
    ],
    "session_id": "session_123",
}


def _tool() -> ToolParams:
    return ToolParams(
        id="web_search",
        name="Web search",
        description="Searches the web.",
        parameters={
            "type": "object",
            "properties": {
                "objective": {"type": "string"},
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["search_queries"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "search_id": {"type": "string"},
                "results": {"type": "array"},
                "session_id": {"type": "string"},
            },
            "required": ["search_id", "results", "session_id"],
        },
    )


def _params(env_kwargs: dict | None = None) -> EnvironmentParams:
    return EnvironmentParams(
        id="parallel",
        env_type="parallel_search",
        tools=[_tool()],
        env_kwargs=env_kwargs,
    )


def test_environment_is_registered_under_parallel_search():
    assert (
        REGISTRY.get("parallel_search", RegistryType.ENVIRONMENT)
        is ParallelSearchEnvironment
    )


@pytest.mark.parametrize(
    ("env_kwargs", "environment_key", "expected_key"),
    [
        ({"api_key": "configured-key"}, None, "configured-key"),
        (None, "environment-key", "environment-key"),
    ],
)
def test_from_params_sends_search_arguments_with_api_key(
    monkeypatch, env_kwargs, environment_key, expected_key
):
    if environment_key is not None:
        monkeypatch.setenv("PARALLEL_API_KEY", environment_key)
    sent: dict[str, object] = {}

    class _Response:
        ok = True

        def json(self) -> JsonValue:
            return _RESPONSE

    def fake_post(self, url, *, json, timeout):
        sent.update(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
                "api_key": self.headers.get("x-api-key"),
            }
        )
        return _Response()

    monkeypatch.setattr(requests.Session, "post", fake_post)
    env = ParallelSearchEnvironment.from_params(_params(env_kwargs))
    arguments = {
        "objective": "Find current Oumi documentation.",
        "search_queries": ["Oumi documentation"],
    }

    result = env.call("web_search", arguments, call_id="call-1", session_id="session-1")

    assert sent == {
        "url": _SEARCH_URL,
        "json": arguments,
        "timeout": 30.0,
        "api_key": expected_key,
    }
    assert result.output == _RESPONSE


def test_from_params_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="PARALLEL_API_KEY"):
        ParallelSearchEnvironment.from_params(_params())


@pytest.mark.parametrize(
    "kwargs",
    [
        ParallelSearchEnvironmentKwargs(endpoint_url=""),
        ParallelSearchEnvironmentKwargs(timeout_seconds=0),
        ParallelSearchEnvironmentKwargs(max_retries=-1),
        ParallelSearchEnvironmentKwargs(api_key_env_varname=""),
    ],
)
def test_kwargs_reject_an_unusable_configuration(kwargs):
    with pytest.raises(ValueError):
        kwargs.finalize_and_validate()


class _StatusClient:
    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    def post_json(self, payload: JsonValue) -> JsonValue:
        raise EndpointStatusError(
            self._status_code, f"Endpoint returned HTTP {self._status_code}."
        )


def test_parallel_validation_error_is_a_tool_error():
    env = ParallelSearchEnvironment(
        _params({"api_key": "unused"}),
        ParallelSearchProtocol(_StatusClient(422)),
    )

    with pytest.raises(ToolError):
        env.call(
            "web_search",
            {"search_queries": ["query"]},
            call_id="call-1",
            session_id="session-1",
        )


def test_parallel_rate_limit_is_an_endpoint_failure():
    env = ParallelSearchEnvironment(
        _params({"api_key": "unused"}),
        ParallelSearchProtocol(_StatusClient(429)),
    )

    with pytest.raises(EndpointCallError):
        env.call(
            "web_search",
            {"search_queries": ["query"]},
            call_id="call-1",
            session_id="session-1",
        )
