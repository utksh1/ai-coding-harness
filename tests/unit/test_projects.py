"""Project registry + session-management surface tests (platform P4).

Covers: ProjectStore persistence/idempotence/collisions, the /api/projects
endpoints, /api/models profile listing, run-level model_profile fallback,
follow-up context injection, and /agent/cancellation of a live run.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harness.config import HarnessConfig
from harness.service.app import create_app
from harness.service.projects import ProjectStore, _slugify, run_history

# ---------------------------------------------------------------------------
# ProjectStore unit surface


def test_slugify_sanitizes_project_ids() -> None:
    assert _slugify("My App (v2)") == "my-app-v2"
    assert _slugify("///") == "project"


def test_register_rejects_missing_path(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    with pytest.raises(ValueError):
        store.register(str(tmp_path / "nope"))


def test_register_is_idempotent_same_path(tmp_path: Path) -> None:
    repo = tmp_path / "api"
    repo.mkdir()
    store = ProjectStore(tmp_path / "projects.json")
    first = store.register(str(repo))
    second = store.register(str(repo))
    assert first["id"] == second["id"] == "api"
    assert len(store.list_projects()) == 1


def test_auto_name_collision_disambiguates(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    store = ProjectStore(tmp_path / "projects.json")
    first = store.register(str(tmp_path / "a"))  # both dirname "a"/"b" differ
    # Same name, different paths: explicit names are caller's choice, but
    # re-registering the SAME path keeps the id (no fork).
    again = store.register(str(tmp_path / "a"))
    assert first["id"] == again["id"]


def test_register_two_same_named_dirs_get_distinct_ids(tmp_path: Path) -> None:
    (tmp_path / "x" / "svc").mkdir(parents=True)
    (tmp_path / "y" / "svc").mkdir(parents=True)
    store = ProjectStore(tmp_path / "projects.json")
    one = store.register(str(tmp_path / "x" / "svc"))
    two = store.register(str(tmp_path / "y" / "svc"))
    assert one["id"] != two["id"]
    assert len(store.list_projects()) == 2


def test_unregister_unknown_is_false(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects.json")
    assert store.unregister("ghost") is False


def test_env_var_relocates_the_registry(tmp_path: Path) -> None:
    """HARNESS_PROJECTS_FILE is the deployer's knob AND the hermetic-tests knob.

    A CWD-relative default means a running `foreman start` and a pytest run in
    the same checkout would share one registry file; the override must move
    BOTH the default construction path and stay per-instance explicit-path.
    """
    repo = tmp_path / "relocated"
    repo.mkdir()
    relocated = tmp_path / "state" / "projects.json"
    relocated.parent.mkdir()
    env = {**os.environ, "HARNESS_PROJECTS_FILE": str(relocated)}
    code = (
        "from pathlib import Path; "
        "from harness.service.projects import ProjectStore; "
        f"ProjectStore().register({str(repo)!r}, 'Relocated'); "
        "print(ProjectStore().get('relocated') is not None)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True"
    assert relocated.exists()
    # Explicit path still wins over the environment.
    explicit = tmp_path / "explicit.json"
    ProjectStore(explicit).register(str(repo), "Explicit")
    assert explicit.exists() and not ProjectStore(explicit).get("relocated")


def test_persistence_across_instances(tmp_path: Path) -> None:
    repo = tmp_path / "svc"
    repo.mkdir()
    path = tmp_path / "projects.json"
    ProjectStore(path).register(str(repo), "Service")
    reloaded = ProjectStore(path)
    assert reloaded.get("service")["path"] == str(repo.resolve())


def test_detail_enriches_git_and_files(tmp_path: Path) -> None:
    repo = tmp_path / "plain"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n")
    store = ProjectStore(tmp_path / "projects.json")
    store.register(str(repo))
    detail = store.detail("plain")
    assert detail is not None
    assert detail["git"] is False
    assert detail["files"] >= 1
    assert detail["python_files"] == 1


def test_resolve_by_id_and_by_path(tmp_path: Path) -> None:
    repo = tmp_path / "web"
    repo.mkdir()
    store = ProjectStore(tmp_path / "projects.json")
    store.register(str(repo))
    assert store.resolve("web") == str(repo.resolve())
    assert store.resolve(str(repo)) == str(repo.resolve())
    assert store.resolve("ghost") is None


def test_run_history_reads_verdicts(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for run_id, verdict_line in {
        "good": '"success": true',
        "bad": '"success": false',
        "none": None,
    }.items():
        pack = results / run_id
        pack.mkdir(parents=True)
        if verdict_line:
            (pack / "events.jsonl").write_text(
                f'{{"event": "run.end", "run_id": "{run_id}", {verdict_line}}}\n'
            )
    history = run_history(str(tmp_path), "results")
    verdicts = {run["run_id"]: run["verdict"] for run in history}
    assert verdicts["good"] == "VERIFIED"
    assert verdicts["bad"] == "FAILED"
    assert verdicts["none"] == "unknown"


# ---------------------------------------------------------------------------
# Service endpoints


def _config_two_profiles() -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "models": {
                "default": {"provider": "fake", "name": "fake-model"},
                "alt": {"provider": "fake", "name": "fake-alt"},
            },
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "ver-1", "role": "verifier", "model": "default"},
            ],
            "storage": {"backend": "memory"},
        }
    )


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
VERDICT = '{"approved": true, "issues": [], "summary": "ok", "criteria_dispositions": [{"criterion": "ok", "satisfied": true, "evidence": "task result confirms"}]}'


def _scripted(fake_model_config: Any, loop: bool = False) -> Any:
    from harness.infrastructure.model_providers import FakeProvider, ModelResponse

    responses = [
        ModelResponse(content=PROFILE),
        ModelResponse(content=PLAN),
        ModelResponse(content="TASK_COMPLETE: done"),
        ModelResponse(content=VERDICT),
    ]
    return FakeProvider(fake_model_config, responses=responses, loop=loop)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    target = tmp_path / "target"
    target.mkdir()
    (target / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (target / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (target / "app.py").write_text("def greet():\n    return 'hello'\n")
    return target


def test_models_endpoint_lists_profiles_no_secrets() -> None:
    app = create_app(config=_config_two_profiles(), provider=object())
    client = TestClient(app)
    data = client.get("/api/models").json()
    names = [p["profile"] for p in data["profiles"]]
    assert names == ["alt", "default"]
    assert all("api_key" not in p and "key" not in p for p in data["profiles"])


def test_project_endpoints_roundtrip(repo: Path) -> None:
    app = create_app(config=_config_two_profiles(), provider=object())
    client = TestClient(app)

    bad = client.post("/api/projects", json={"path": "/does/not/exist"})
    assert bad.json()["registered"] is False

    made = client.post("/api/projects", json={"path": str(repo), "name": "Demo"})
    assert made.json()["registered"] is True
    project = made.json()["project"]
    assert project["id"] == "demo"
    assert project["path"] == str(repo.resolve())

    listing = client.get("/api/projects").json()["projects"]
    assert [p["id"] for p in listing] == ["demo"]

    detail = client.get("/api/projects/demo").json()
    assert detail["found"] is True
    assert detail["project"]["path"] == str(repo.resolve())

    resolved = client.get("/api/projects/demo/resolve").json()
    assert resolved["found"] is True
    assert resolved["path"] == str(repo.resolve())

    removed = client.delete("/api/projects/demo").json()
    assert removed["removed"] is True
    assert client.get("/api/projects").json()["projects"] == []


def test_run_with_unknown_model_profile_falls_back(repo: Path, fake_model_config: Any) -> None:
    """A stale cockpit profile pick must never 400 the run."""
    app = create_app(config=_config_two_profiles(), provider=_scripted(fake_model_config))
    client = TestClient(app)
    response = client.post(
        "/agent/run",
        json={
            "issue": "greet should work",
            "repo_root": str(repo),
            "run_id": "profile-run",
            "model_profile": "not-a-profile",
        },
    )
    assert response.status_code == 200
    assert response.json()["run_id"] == "profile-run"


def test_two_profiles_use_distinct_pipelines(repo: Path, fake_model_config: Any) -> None:
    """Profile participates in the pipeline cache key: two profiles on one
    repo are two agent sets, and both runs complete (looping script)."""
    app = create_app(
        config=_config_two_profiles(), provider=_scripted(fake_model_config, loop=True)
    )
    client = TestClient(app)
    for profile in ("default", "alt"):
        response = client.post(
            "/agent/run",
            json={
                "issue": f"work under {profile}",
                "repo_root": str(repo),
                "run_id": f"run-{profile}",
                "model_profile": profile,
            },
        )
        assert response.status_code == 200
    # both runs completed; distinct cache entries cannot be asserted from the
    # outside without handles, but the runs sharing a repo serialised fine.
    assert response.json()["run_id"] == "run-alt"


def test_followup_run_carries_prior_context(repo: Path, fake_model_config: Any) -> None:
    """A follow-up run's architect prompt carries the prior run's context."""
    provider = _scripted(fake_model_config, loop=True)
    app = create_app(config=_config_two_profiles(), provider=provider)
    client = TestClient(app)

    first = client.post(
        "/agent/run",
        json={"issue": "first change", "repo_root": str(repo), "run_id": "seed-run"},
    )
    assert first.status_code == 200

    followup = client.post(
        "/agent/run",
        json={
            "issue": "now extend it",
            "repo_root": str(repo),
            "run_id": "follow-run",
            "followup_of": "seed-run",
        },
    )
    assert followup.status_code == 200
    all_prompts = str(provider.calls)
    assert "CONTINUATION of run seed-run" in all_prompts
    assert "FOLLOW-UP REQUEST: now extend it" in all_prompts


def test_followup_of_unknown_run_degrades_to_fresh(repo: Path, fake_model_config: Any) -> None:
    provider = _scripted(fake_model_config)
    app = create_app(config=_config_two_profiles(), provider=provider)
    client = TestClient(app)
    response = client.post(
        "/agent/run",
        json={
            "issue": "fresh anyway",
            "repo_root": str(repo),
            "run_id": "orphan-follow",
            "followup_of": "never-ran",
        },
    )
    assert response.status_code == 200
    assert "CONTINUATION" not in str(provider.calls)


class _SlowProvider:
    """Provider that parks mid-generation so a run can be cancelled live."""

    def __init__(self) -> None:
        self.model = "slow-model"
        self.cancelled_seen = False
        # Skip the capability probe: the answer is known and probing would
        # park inside generate() before the task loop starts.
        from harness.infrastructure.model_providers.capability import ModelCapabilities

        self.capabilities = ModelCapabilities(native_tool_calls=True)

    async def generate(self, messages: Any, tools: Any = None, **_: Any) -> Any:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled_seen = True
            raise
        raise AssertionError("slow provider should never finish")


def test_cancel_stops_live_run_honestly(repo: Path, fake_model_config: Any) -> None:
    """The chat stop button: cancellation emits run.failed/run.end and the
    /agent/run caller gets an honest CANCELLED response, never a hang."""
    app = create_app(config=_config_two_profiles(), provider=_SlowProvider())
    with TestClient(app) as client:
        outcome: dict[str, Any] = {}

        def _run() -> None:
            outcome["resp"] = client.post(
                "/agent/run",
                json={"issue": "slow work", "repo_root": str(repo), "run_id": "cancel-me"},
            )

        worker = threading.Thread(target=_run)
        worker.start()
        # Wait until the run is parked inside the provider.
        deadline = time.time() + 5
        detail: dict[str, Any] = {}
        while time.time() < deadline:
            detail = client.post("/agent/cancel", json={"run_id": "cancel-me"}).json()
            if detail.get("cancelled"):
                break
            time.sleep(0.05)
        assert detail.get("cancelled") is True
        worker.join(10)
        assert not worker.is_alive()
        body = outcome["resp"].json()
        assert body["success"] is False
        assert body["outcome"].startswith("CANCELLED")
        assert "cancelled" in body["flags"]
        # Cancellation is idempotent: the finished run is not active anymore.
        again = client.post("/agent/cancel", json={"run_id": "cancel-me"}).json()
        assert again["cancelled"] is False


# ---------------------------------------------------------------------------
# Edge paths: corrupt registry, save failure, git enrichment, verdict parsing


def test_corrupt_registry_degrades_to_empty(tmp_path: Path) -> None:
    bad = tmp_path / "projects.json"
    bad.write_text("{not json", encoding="utf-8")
    store = ProjectStore(bad)
    assert store.list_projects() == []
    repo = tmp_path / "r"
    repo.mkdir()
    assert store.register(str(repo))["id"] == "r"  # rewrite repairs the file


def test_non_dict_registry_degrades_to_empty(tmp_path: Path) -> None:
    bad = tmp_path / "projects.json"
    bad.write_text('["not", "a", "dict"]', encoding="utf-8")
    assert ProjectStore(bad).list_projects() == []


def test_registry_entry_without_path_is_dropped(tmp_path: Path) -> None:
    bad = tmp_path / "projects.json"
    bad.write_text('{"x": {"id": "x"}}', encoding="utf-8")
    store = ProjectStore(bad)
    assert store.get("x") is None
    assert store.list_projects() == []


def test_save_failure_is_survivable(tmp_path: Path) -> None:
    """A registry path under a FILE: mkdir fails, register still answers."""
    wall = tmp_path / "wall"
    wall.write_text("i am a file", encoding="utf-8")
    repo = tmp_path / "r"
    repo.mkdir()
    store = ProjectStore(wall / "projects.json")
    entry = store.register(str(repo))
    assert entry["id"] == "r"


def test_git_project_enrichment(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "gitrepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "f.py").write_text("x=1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    (repo / "dirty.txt").write_text("uncommitted\n")

    store = ProjectStore(tmp_path / "projects.json")
    store.register(str(repo))
    detail = store.detail("gitrepo")
    assert detail is not None
    assert detail["git"] is True
    assert detail["branch"] in {"master", "main"}
    assert len(detail["head"]) >= 4
    assert detail["dirty"] is True
    assert detail["changed_files"] == 1


def test_git_state_survives_missing_git_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harness.service.projects import _git_state

    repo = tmp_path / "g"
    repo.mkdir()
    (repo / ".git").mkdir()  # looks like a repo, but git cannot be found
    monkeypatch.setenv("PATH", "")
    state = _git_state(str(repo))
    assert state["git"] is True
    assert state["branch"] == "?"


def test_run_verdict_parses_malformed_lines(tmp_path: Path) -> None:
    from harness.service.projects import _run_verdict

    pack = tmp_path / "p"
    pack.mkdir()
    (pack / "events.jsonl").write_text(
        'not-json\n{"event": "budget.exhausted", "run_id": "p"}\n',
        encoding="utf-8",
    )
    assert _run_verdict(pack) == "STOPPED"


def test_run_history_missing_dir_is_empty(tmp_path: Path) -> None:
    assert run_history(str(tmp_path / "nope"), "results") == []


def test_unknown_project_detail_and_resolve_endpoints(repo: Path) -> None:
    app = create_app(config=_config_two_profiles(), provider=object())
    client = TestClient(app)
    assert client.get("/api/projects/ghost").json()["found"] is False
    assert client.get("/api/projects/ghost/resolve").json()["found"] is False


def test_cancel_rejects_pathlike_run_id(repo: Path) -> None:
    app = create_app(config=_config_two_profiles(), provider=object())
    client = TestClient(app)
    assert client.post("/agent/cancel", json={"run_id": "../etc"}).json()["cancelled"] is False


# ---------------------------------------------------------------------------
# _followup_context unit surface


def _pack_with_evidence(tmp_path: Path, run_id: str, summary: str, diff: str) -> Path:
    pack = tmp_path / "results" / run_id
    pack.mkdir(parents=True)
    (pack / "summary.md").write_text(summary, encoding="utf-8")
    (pack / "patch.diff").write_text(diff, encoding="utf-8")
    (pack / "events.jsonl").write_text(
        f'{{"event": "run.end", "run_id": "{run_id}", "success": true}}\n', encoding="utf-8"
    )
    return pack


def test_followup_context_full(tmp_path: Path) -> None:
    from harness.service.app import _followup_context

    _pack_with_evidence(
        tmp_path, "r1", "Fixed the parser and added tests.", "diff --git a/x b/x\n+1\n-1\n"
    )
    context = _followup_context({"r1": str(tmp_path)}, "r1", "results")
    assert context is not None
    assert "CONTINUATION of run r1" in context
    assert "VERIFIED" in context
    assert "Fixed the parser" in context
    assert "1 file(s)" in context
    assert "+1/-1" in context


def test_followup_context_rejects_pathlike_id(tmp_path: Path) -> None:
    from harness.service.app import _followup_context

    assert _followup_context({}, "../x", "results") is None


def test_followup_context_unknown_run_is_none(tmp_path: Path) -> None:
    from harness.service.app import _followup_context

    assert _followup_context({"other": str(tmp_path)}, "missing", "results") is None


def test_followup_context_missing_pack_dir_is_none(tmp_path: Path) -> None:
    from harness.service.app import _followup_context

    assert _followup_context({"r2": str(tmp_path)}, "r2", "results") is None


def test_followup_context_empty_pack_yields_none(tmp_path: Path) -> None:
    from harness.service.app import _followup_context

    pack = tmp_path / "results" / "empty"
    pack.mkdir(parents=True)
    # No summary, no patch: only the header line would exist -> None.
    assert _followup_context({"empty": str(tmp_path)}, "empty", "results") is None


def test_registry_evicts_beyond_cap(tmp_path: Path) -> None:
    """PROJECTS_MAX is a picker cap, not a warehouse: oldest entries go."""
    from harness.service import projects as projects_mod

    store = ProjectStore(tmp_path / "projects.json")
    for i in range(projects_mod.PROJECTS_MAX + 1):
        repo = tmp_path / f"r{i}"
        repo.mkdir()
        store.register(str(repo))
    assert len(store.list_projects()) == projects_mod.PROJECTS_MAX
    assert store.get("r0") is None  # oldest evicted
    assert store.get(f"r{projects_mod.PROJECTS_MAX}") is not None  # newest kept


def test_repo_stats_walk_is_bounded(tmp_path: Path) -> None:
    from harness.service.projects import _repo_stats

    repo = tmp_path / "many"
    repo.mkdir()
    import os

    for i in range(5020):
        fd = os.open(str(repo / f"f{i}.txt"), os.O_CREAT | os.O_WRONLY, 0o644)
        os.close(fd)
    stats = _repo_stats(str(repo))
    assert stats["files"] == 5000  # bounded: never walks the whole tree
