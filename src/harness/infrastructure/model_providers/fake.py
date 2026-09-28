"""Fake provider: scripted, offline, deterministic (the test-suite backbone).

`make test` runs the whole harness against this provider without any API key
or network access; the live eval swaps in a real provider via configuration.
"""

from __future__ import annotations

from typing import Any

from harness.infrastructure.model_providers.base import (
    ModelProvider,
    ModelResponse,
    ToolCall,
    _approx_tokens,
)


class FakeProvider(ModelProvider):
    """Replays a scripted sequence of responses.

    Entries may be `ModelResponse`s (replayed verbatim) or `Exception`s
    (raised once, useful for retry/recovery tests). Exhausting the script
    fails loudly rather than hallucinating further responses.
    """

    def __init__(
        self,
        config: Any,
        responses: list[Any],
        client: Any = None,
        loop: bool = False,
    ) -> None:
        super().__init__(config, client)
        self._initial_responses = [
            r.model_copy() if hasattr(r, "model_copy") else r for r in responses
        ]
        self._responses = list(responses)
        self._loop = loop
        self.calls: list[dict[str, Any]] = []
        from harness.infrastructure.model_providers.capability import ModelCapabilities

        # Scripted models speak native tool calls unless a test overrides this.
        self.capabilities = ModelCapabilities(native_tool_calls=True)

    def _resolve_api_key(self) -> str:
        return "fake-key"

    def _endpoint(self) -> str:
        return "fake://generate"

    def _headers(self, api_key: str) -> dict[str, str]:
        return {}

    def _payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        return {"messages": messages, "tools": tools or []}

    def _parse_response(
        self, data: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelResponse:  # pragma: no cover - never reached (generate is overridden)
        raise NotImplementedError

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "tools": tools, "overrides": overrides})
        if not self._responses and self._loop and self._initial_responses:
            self._responses = [
                r.model_copy() if hasattr(r, "model_copy") else r for r in self._initial_responses
            ]
        if not self._responses:
            if self._loop:
                return ModelResponse(
                    content="TASK_COMPLETE: demo step finished",
                    prompt_tokens=10,
                    completion_tokens=5,
                )
            msg = (
                f"FakeProvider script exhausted after {len(self.calls) - 1} calls; "
                "extend the script in the test"
            )
            raise RuntimeError(msg)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        prompt_estimate = sum(_approx_tokens(str(m.get("content") or "")) for m in messages)
        item.prompt_tokens = item.prompt_tokens or prompt_estimate
        item.completion_tokens = item.completion_tokens or _approx_tokens(item.content)
        return item


def build_demo_provider(config: Any) -> FakeProvider:
    """Credential-free demo: a canned end-to-end script for any issue.

    The script performs a REAL edit (filesystem_write of demo_output.py) so
    the run carries an actual diff - the no-op gate would (correctly) fail
    a demo that only claims completion in prose. The pipeline labels the
    resulting evidence DEMO; never use during evaluation - this exists so
    judges can see the full flow offline.
    """
    import json

    responses = [
        ModelResponse(
            content=json.dumps(
                {
                    "languages": ["Python"],
                    "frameworks": [],
                    "test_framework": "pytest",
                    "build_system": "pyproject.toml",
                    "conventions": [],
                    "notes": "demo mode (scripted)",
                }
            )
        ),
        ModelResponse(
            content=json.dumps(
                {
                    "issue_summary": "demo run",
                    "complexity": 2,
                    "subtasks": [
                        {
                            "id": "st-1",
                            "title": "write the demo marker module",
                            "description": "demo-mode placeholder change",
                            "specialty": "refactoring",
                            "complexity": 1,
                            "files": ["demo_output.py"],
                            "acceptance_criteria": ["demo completes"],
                            "depends_on": [],
                        }
                    ],
                    "risks": ["demo mode produces no real changes"],
                    "needs_collaboration": False,
                }
            )
        ),
        ModelResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="demo-write",
                    name="filesystem_write",
                    arguments={
                        "path": "demo_output.py",
                        "content": (
                            '"""Demo-mode marker written by the scripted specialist."""\n'
                            'DEMO_MARKER = "foreman-demo"\n'
                        ),
                        "mode": "create",
                    },
                )
            ],
        ),
        ModelResponse(content="TASK_COMPLETE: demo-mode specialist wrote demo_output.py (scripted)"),
        ModelResponse(
            content=json.dumps(
                {
                    "approved": True,
                    "issues": [],
                    "summary": "demo verdict (scripted)",
                    "criteria_dispositions": [
                        {
                            "criterion": "demo completes",
                            "satisfied": True,
                            "evidence": "demo_output.py written (scripted)",
                        }
                    ],
                }
            )
        ),
    ]
    return _DemoProvider(config, responses)


class _DemoProvider(FakeProvider):
    """Demo replay that tails the script instead of wrapping it.

    The final review makes one structured call per diff chunk; a demo repo
    with a dirty tree (e.g. leftover ``.harness/`` artifacts) can exceed the
    scripted call count. Wrapping to the FIRST response would feed a verdict
    parser the repository PROFILE (live finding: ReviewVerdict crashed on
    ``{'languages': ...}``). Tailing the LAST scripted response - the
    verdict - keeps every extra call semantically valid.
    """

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> ModelResponse:
        if not self._responses and self._initial_responses:
            self._responses = [self._initial_responses[-1].model_copy()]
        return await super().generate(messages, tools, **overrides)
