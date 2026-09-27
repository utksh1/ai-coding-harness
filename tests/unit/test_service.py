"""Orchestrator service + Redis publisher tests (platform P1/P3, #70/#72)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.config import HarnessConfig
from harness.service.app import create_app
from harness.service.events import RedisEventPublisher


class FakeRedis:
    """Minimal redis-like client recording publishes."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1


@pytest.fixture
def demo_repo(tmp_path: Path) -> Path:
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "target" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "target" / "app.py").write_text("def greet():\n    return 'hello'\n")
    (tmp_path / "harness.yaml").write_text(
        "models:\n  default:\n    provider: fake\n    name: fake-model\n"
        "agents:\n"
        "  - agent_id: arch-1\n    role: architect\n    model: default\n"
        "  - agent_id: mgr-1\n    role: manager\n    model: default\n"
        "  - agent_id: ver-1\n    role: verifier\n    model: default\n"
        "storage:\n  backend: memory\n"
    )
    return tmp_path


PROFILE = (
    '{"languages": ["Python"], "frameworks": [], "test_framework": "pytest", '
    '"build_system": "pyproject.toml", "conventions": [], "notes": ""}'
)
PLAN = (
    '{"issue_summary": "s", "complexity": 2, "subtasks": [{"id": "st-1", '
    '"title": "t", "description": "d", "specialty": "verification", '
    '"complexity": 1, "files": ["app.py"], "acceptance_criteria": ["ok"], '
    '"depends_on": []}], "risks": [], "needs_collaboration": false}'
)
VERDICT = '{"approved": true, "issues": [], "summary": "ok"}'


def _service_config() -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "models": {"default": {"provider": "fake", "name": "fake-model"}},
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "ver-1", "role": "verifier", "model": "default"},
            ],
            "storage": {"backend": "memory"},
        }
    )


def _scripted_provider(fake_model_config):
    from harness.infrastructure.model_providers import FakeProvider, ModelResponse

    return FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE),
            ModelResponse(content=PLAN),
            ModelResponse(content="TASK_COMPLETE: done"),
            ModelResponse(content=VERDICT),
        ],
    )


def test_health() -> None:
    app = create_app(config=_service_config(), provider=object())
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_agents_endpoint_lists_configured_roster() -> None:
    app = create_app(config=_service_config(), provider=object())
    client = TestClient(app)
    response = client.get("/api/agents")
    assert response.status_code == 200
    agents = response.json()["agents"]
    assert {a["agent_id"] for a in agents} == {"arch-1", "mgr-1", "ver-1"}
    assert {a["role"] for a in agents} == {"architect", "manager", "verifier"}


def test_run_endpoint_honors_supplied_run_id(demo_repo: Path, fake_model_config) -> None:
    """The gateway threads ONE run id: events, response, and evidence share it."""
    fake_redis = FakeRedis()
    app = create_app(
        config=_service_config(),
        provider=_scripted_provider(fake_model_config),
        redis_client=fake_redis,
    )
    client = TestClient(app)
    response = client.post(
        "/agent/run",
        json={
            "issue": "greet should work",
            "repo_root": str(demo_repo / "target"),
            "run_id": "gateway-42",
        },
    )
    body = response.json()
    assert body["run_id"] == "gateway-42"
    assert (demo_repo / "target" / "results" / "gateway-42").is_dir()
    # every streamed event carries the same id
    payloads = [json.loads(message) for _, message in fake_redis.published]
    assert payloads and all(p["run_id"] == "gateway-42" for p in payloads)
    # the new cockpit events stream too: verification stages + token usage
    kinds = [p["event"] for p in payloads]
    assert "verification.stage" in kinds
    assert "tokens.usage" in kinds
    stage = next(p for p in payloads if p["event"] == "verification.stage")
    assert stage["stage"] == "1-integrity" and stage["passed"] is True
    usage = next(p for p in payloads if p["event"] == "tokens.usage")
    assert usage["usage"]["total_tokens"] >= 0


def test_run_endpoint_rejects_unsafe_run_id(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/run",
        json={"issue": "x", "repo_root": ".", "run_id": "../escape"},
    )
    body = response.json()
    assert body["success"] is False and "invalid run_id" in body["error"]


def test_evidence_files_endpoint(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    client.post(
        "/agent/run",
        json={"issue": "x", "repo_root": str(demo_repo / "target"), "run_id": "evd-1"},
    )
    response = client.get(
        "/api/evidence/evd-1/files", params={"repo_root": str(demo_repo / "target")}
    )
    assert response.json()["found"] is True
    assert "patch.diff" in response.json()["files"]

    missing = client.get(
        "/api/evidence/nope/files", params={"repo_root": str(demo_repo / "target")}
    )
    assert missing.json()["found"] is False

    # path-like run ids: the router normalizes "../", but a backslash
    # survives routing and must be refused by the endpoint guard
    unsafe = client.get("/api/evidence/a\\b/files")
    assert unsafe.json()["found"] is False and "invalid run_id" in unsafe.json()["error"]
    from harness.service.app import _safe_run_id

    assert _safe_run_id("../escape") is False
    assert _safe_run_id("evd-1") is True
    assert _safe_run_id("") is False


def test_evidence_file_endpoint(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    client.post(
        "/agent/run",
        json={"issue": "x", "repo_root": str(demo_repo / "target"), "run_id": "evd-2"},
    )
    response = client.get(
        "/api/evidence/evd-2/file/summary.md", params={"repo_root": str(demo_repo / "target")}
    )
    body = response.json()
    assert body["found"] is True and "VERIFIED" in body["content"]

    # a whitelisted name that this run never wrote: baseline.json is removed
    # to simulate a pack without it (the guard branch needs a real miss)
    baseline_path = demo_repo / "target" / "results" / "evd-2" / "baseline.json"
    if baseline_path.exists():
        baseline_path.unlink()
    missing_file = client.get(
        "/api/evidence/evd-2/file/baseline.json", params={"repo_root": str(demo_repo / "target")}
    )
    assert missing_file.json()["found"] is False

    forbidden = client.get(
        "/api/evidence/evd-2/file/trace.jsonl", params={"repo_root": str(demo_repo / "target")}
    )
    assert forbidden.json()["found"] is False and "not exposed" in forbidden.json()["error"]

    missing_run = client.get(
        "/api/evidence/ghost/file/summary.md", params={"repo_root": str(demo_repo / "target")}
    )
    assert missing_run.json()["found"] is False


def test_evidence_resolves_repo_from_run_registry(demo_repo: Path, fake_model_config) -> None:
    """The gateway's evidence proxy passes no repo_root; the orchestrator
    remembers where each run executed and resolves the pack from there."""
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    client.post(
        "/agent/run",
        json={"issue": "x", "repo_root": str(demo_repo / "target"), "run_id": "evd-reg"},
    )
    # no repo_root query param at all - the registry must supply it
    listing = client.get("/api/evidence/evd-reg/files")
    assert listing.json()["found"] is True
    assert "patch.diff" in listing.json()["files"]

    content = client.get("/api/evidence/evd-reg/file/summary.md")
    assert content.json()["found"] is True and "VERIFIED" in content.json()["content"]

    # unknown run + no repo_root: falls back to "." and misses honestly
    unknown = client.get("/api/evidence/unknown-run/file/summary.md")
    assert unknown.json()["found"] is False


def test_run_root_registry_persists_across_restart(tmp_path: Path, monkeypatch) -> None:
    """Evidence links survive an orchestrator restart (the cockpit's Diff
    tab fetches patch.diff long after the run finished)."""
    from harness.service import app as service_app

    target = tmp_path / "roots" / "run-roots.json"
    monkeypatch.setattr(service_app, "RUN_ROOTS_FILE", target)

    service_app._save_run_roots({"run-a": "/repo/a", "run-b": "/repo/b"})
    assert target.exists()
    assert service_app._load_run_roots() == {"run-a": "/repo/a", "run-b": "/repo/b"}

    # a fresh process (new app instance) reads the same registry
    registry = service_app._load_run_roots()
    service_app._remember_run_root(registry, "run-c", "/repo/c")
    service_app._save_run_roots(registry)
    assert service_app._load_run_roots()["run-c"] == "/repo/c"


def test_run_roots_load_tolerates_damage(tmp_path: Path, monkeypatch) -> None:
    from harness.service import app as service_app

    target = tmp_path / "run-roots.json"
    monkeypatch.setattr(service_app, "RUN_ROOTS_FILE", target)
    # not JSON
    target.write_text("]]not json[[")
    assert service_app._load_run_roots() == {}
    # JSON but not an object
    target.write_text('["a", "b"]')
    assert service_app._load_run_roots() == {}
    # absent file
    target.unlink()
    assert service_app._load_run_roots() == {}


def test_run_roots_save_never_raises_on_unwritable(tmp_path: Path, monkeypatch) -> None:
    from harness.service import app as service_app

    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the directory should be")
    monkeypatch.setattr(service_app, "RUN_ROOTS_FILE", blocked / "run-roots.json")
    service_app._save_run_roots({"run-a": "/repo/a"})  # must not raise


def test_run_root_registry_evicts_oldest() -> None:
    from harness.service.app import _remember_run_root

    registry: dict[str, str] = {}
    for i in range(4):
        _remember_run_root(registry, f"run-{i}", f"/repo-{i}", cap=3)
    assert list(registry) == ["run-1", "run-2", "run-3"]  # run-0 evicted
    # re-recording an existing id refreshes without growth
    _remember_run_root(registry, "run-1", "/repo-1b", cap=3)
    assert list(registry) == ["run-2", "run-3", "run-1"]
    assert registry["run-1"] == "/repo-1b"


def test_analyze_endpoint(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/architect/analyze", json={"repo_root": str(demo_repo / "target")}
    )
    assert response.status_code == 200
    assert response.json()["profile"]["test_framework"] == "pytest"


def test_decompose_endpoint(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/architect/decompose",
        json={"issue": "greet works", "repo_root": str(demo_repo / "target")},
    )
    assert response.status_code == 200
    assert response.json()["plan"]["subtasks"][0]["id"] == "st-1"


def test_assign_and_status_endpoints(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/manager/assign",
        json={
            "task": {"id": "t-1", "title": "x", "description": "y"},
            "agent_id": "ver-1",
        },
    )
    assert response.status_code == 200 and response.json()["assigned"] is True
    # routing is bookkeeping: the manager reports idle until it executes
    status = client.get("/agent/status/mgr-1").json()
    assert status["known"] is True and status["status"] == "idle"
    unknown = client.get("/agent/status/ghost").json()
    assert unknown["known"] is False


def test_specialist_execute_endpoint(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/specialist/execute",
        json={
            "task": {"id": "t-9", "title": "t", "description": "d"},
        },
    )
    assert response.status_code == 200
    assert response.json()["result"]["task_id"] == "t-9"


def test_run_endpoint_publishes_redis_events(demo_repo: Path, fake_model_config) -> None:
    fake_redis = FakeRedis()
    app = create_app(
        config=_service_config(),
        provider=_scripted_provider(fake_model_config),
        redis_client=fake_redis,
    )
    client = TestClient(app)
    response = client.post(
        "/agent/run",
        json={
            "issue": "greet should work",
            "repo_root": str(demo_repo / "target"),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True and "VERIFIED" in body["outcome"]
    channels = {channel for channel, _ in fake_redis.published}
    assert any(channel.startswith("harness.events.") for channel in channels)
    payloads = [json.loads(message) for _, message in fake_redis.published]
    kinds = [p["event"] for p in payloads]
    assert "run.start" in kinds and "run.end" in kinds


def test_run_demo_mode_flag(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/run",
        json={
            "issue": "demo",
            "repo_root": str(demo_repo / "target"),
            "demo_mode": True,
        },
    )
    assert response.json()["success"] is True


class _ExplodingProvider:
    """A provider whose generate raises: transport death mid-run."""

    async def generate(self, *args, **kwargs):  # pragma: no cover - raises
        raise RuntimeError("model request failed after 4 attempts: 429")

    async def aclose(self) -> None:
        return None


def test_run_endpoint_fails_gracefully_on_transport_death(
    demo_repo: Path, fake_model_config
) -> None:
    """A dead model transport must end in an honest failure, never a 500 hang."""
    fake_redis = FakeRedis()
    app = create_app(
        config=_service_config(),
        provider=_ExplodingProvider(),
        redis_client=fake_redis,
    )
    client = TestClient(app)
    response = client.post(
        "/agent/run",
        json={"issue": "x", "repo_root": str(demo_repo / "target"), "run_id": "boom-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["run_id"] == "boom-1"
    assert "FAILED" in body["outcome"] and "transport" in body["flags"][0]
    payloads = [json.loads(message) for _, message in fake_redis.published]
    kinds = [p["event"] for p in payloads]
    assert "run.failed" in kinds and "run.end" in kinds
    end = next(p for p in payloads if p["event"] == "run.end")
    assert end["success"] is False and end["run_id"] == "boom-1"


def test_evidence_latest_endpoint(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    client.post("/agent/run", json={"issue": "x", "repo_root": str(demo_repo / "target")})
    response = client.get("/evidence/latest", params={"repo_root": str(demo_repo / "target")})
    assert response.json()["found"] is True
    missing = client.get("/evidence/latest", params={"repo_root": str(demo_repo / "nope")})
    assert missing.json()["found"] is False


# -- publisher unit behavior (#72) ------------------------------------------------
def test_publisher_without_redis_is_noop() -> None:
    publisher = RedisEventPublisher()  # no client, no REDIS_URL
    publisher.publish("run-1", {"event": "x"})  # must not raise
    publisher.sink_for("run-1")({"event": "y"})


def test_publisher_publishes_json_envelope(fake_model_config) -> None:
    fake = FakeRedis()
    publisher = RedisEventPublisher(client=fake)
    publisher.sink_for("run-9")({"event": "run.start", "run_id": "run-9"})
    channel, message = fake.published[0]
    assert channel == "harness.events.run-9"
    assert json.loads(message)["event"] == "run.start"


def test_publisher_swallows_client_errors(fake_model_config) -> None:
    class ExplodingRedis:
        def publish(self, channel: str, message: str) -> int:
            raise RuntimeError("redis down")

    publisher = RedisEventPublisher(client=ExplodingRedis())
    publisher.publish("run-1", {"event": "x"})  # must not raise


def test_evidence_sink_exceptions_never_break_a_run(tmp_path: Path) -> None:
    """The event sink is best-effort: a raising sink is swallowed."""
    from harness.engine.evidence import EvidencePack

    def exploding(event: dict) -> None:
        raise RuntimeError("redis down")

    pack = EvidencePack(tmp_path, "run-sink", event_sink=exploding)
    pack.trace({"event": "x"})  # must not raise
    assert pack.read_trace()[0]["event"] == "x"


def test_config_lazy_load_when_none(monkeypatch, tmp_path: Path) -> None:
    """create_app(config=None) loads harness.yaml from cwd on first use."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "harness.yaml").write_text(
        "models:\n  default:\n    provider: fake\n    name: fake-model\n"
    )
    app = create_app(config=None, provider=object())
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    # the lazy config load triggers on any endpoint that needs the pipeline
    status = client.get("/agent/status/ver-1").json()
    assert status["known"] is False  # lazy config has no agents; load proven


def test_architect_fallback_when_config_has_no_architect(
    demo_repo: Path, fake_model_config
) -> None:
    config = HarnessConfig.model_validate(
        {
            "models": {"default": {"provider": "fake", "name": "fake-model"}},
            "agents": [{"agent_id": "ver-1", "role": "verifier", "model": "default"}],
            "storage": {"backend": "memory"},
        }
    )
    app = create_app(config=config, provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/architect/analyze", json={"repo_root": str(demo_repo / "target")}
    )
    assert response.status_code == 200
    assert response.json()["profile"]["test_framework"] == "pytest"


def test_assign_without_manager_reports_absent(fake_model_config) -> None:
    config = HarnessConfig.model_validate(
        {
            "models": {"default": {"provider": "fake", "name": "fake-model"}},
            "agents": [{"agent_id": "ver-1", "role": "verifier", "model": "default"}],
            "storage": {"backend": "memory"},
        }
    )
    app = create_app(config=config, provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/manager/assign",
        json={
            "task": {"id": "t-1", "title": "x", "description": "y"},
            "agent_id": "ver-1",
        },
    )
    assert response.json()["assigned"] is False
    assert "no manager configured" in response.json()["detail"]


def test_evidence_latest_without_runs(demo_repo: Path, fake_model_config) -> None:
    app = create_app(config=_service_config(), provider=_scripted_provider(fake_model_config))
    client = TestClient(app)
    (demo_repo / "target" / "results").mkdir()
    response = client.get("/evidence/latest", params={"repo_root": str(demo_repo / "target")})
    assert response.json()["found"] is False


def test_publisher_lazy_client_from_url(monkeypatch) -> None:
    """REDIS_URL set + no client: a real client is created lazily (faked here)."""
    import sys
    import types

    fake_module = types.ModuleType("redis")

    class FakeRedisClient:
        def __init__(self, url: str, decode_responses: bool = False) -> None:
            FakeRedisClient.url = url

        def publish(self, channel: str, message: str) -> int:
            return 1

    fake_module.Redis = types.SimpleNamespace(from_url=FakeRedisClient)
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    publisher = RedisEventPublisher()  # no client -> lazy from REDIS_URL
    publisher.publish("run-lazy", {"event": "x"})
    assert FakeRedisClient.url == "redis://localhost:6379/0"


def test_reproduction_node_normalization() -> None:
    """Architects emit node ids, prefixed ids, and full command lines."""
    from harness.verification.pipeline import _reproduction_node

    assert _reproduction_node("pytest test_app.py::test_add") == "test_app.py::test_add"
    assert _reproduction_node("test_app.py::test_add") == "test_app.py::test_add"
    # full command lines with trailing flags (the real-model regression:
    # 'pytest test.py -v' used to become the literal path 'test.py -v')
    assert _reproduction_node("pytest test_calculator.py -v") == "test_calculator.py"
    assert _reproduction_node("pytest -v test_calculator.py") == "test_calculator.py"
    assert _reproduction_node("python -m pytest test_app.py::test_add") == "test_app.py::test_add"
    assert _reproduction_node("python3 -m pytest -q test_app.py") == "test_app.py"
    # value-taking flags: flag AND its value must be skipped
    assert _reproduction_node("pytest -k add test_app.py") == "test_app.py"
    assert _reproduction_node("pytest test_app.py --tb=short") == "test_app.py"
    assert _reproduction_node("pytest test_app.py --junitxml out.xml") == "test_app.py"
    # degenerate inputs
    assert _reproduction_node("pytest") == ""
    assert _reproduction_node("pytest -v") == ""
    assert _reproduction_node("") == ""
    assert _reproduction_node("   ") == ""


async def _noop():
    from harness.verification.pipeline import StageResult

    return StageResult("x", False, "fail")
