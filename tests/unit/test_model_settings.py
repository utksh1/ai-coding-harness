"""Settings-based model management (platform P6).

The ModelStore overlays harness.yaml: profiles can be added, edited, and
removed at runtime; every agent can be re-bound to any profile individually.
The effective config is what pipelines build from, so a settings change is
live on the NEXT run without an orchestrator restart. Keys are env var
NAMES, never values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harness.config import HarnessConfig, ModelConfig
from harness.service.app import create_app
from harness.service.models import MAX_PROFILES, ModelStore


def _base_config() -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "models": {
                "default": {"provider": "fake", "name": "fake-model"},
                "luna": {
                    "provider": "openai-compatible",
                    "name": "gpt-5.6-luna",
                    "api_key_env": "LUNA_API_KEY",
                    "base_url": "https://api.example/v1",
                },
            },
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "impl-1", "role": "implementer", "model": "luna"},
            ],
            "storage": {"backend": "memory"},
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> ModelStore:
    return ModelStore(tmp_path / "models.json")


# -- ModelStore unit ------------------------------------------------------


def test_apply_is_identity_with_empty_overlay(store: ModelStore) -> None:
    cfg = _base_config()
    assert store.apply(cfg) is cfg


def test_upsert_adds_profile_and_apply_uses_it(store: ModelStore) -> None:
    ok, _ = store.upsert_profile(
        "gemini-flash",
        {"provider": "google", "name": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
        _base_config().models,
    )
    assert ok
    effective = store.apply(_base_config())
    assert "gemini-flash" in effective.models
    assert effective.models["gemini-flash"].provider == "google"
    # yaml profiles untouched
    assert effective.models["luna"].name == "gpt-5.6-luna"


def test_upsert_rejects_invalid_name(store: ModelStore) -> None:
    for bad in ("Gemini", "with space", "x" * 41, "", "-leading"):
        ok, error = store.upsert_profile(bad, {}, _base_config().models)
        assert not ok, bad
        assert "invalid profile name" in error


def test_upsert_rejects_invalid_config(store: ModelStore) -> None:
    ok, error = store.upsert_profile("bad-temp", {"temperature": 9.0}, _base_config().models)
    assert not ok and "invalid model config" in error
    # unknown provider: pydantic Literal rejects it inside ModelConfig validation
    ok, error = store.upsert_profile("bad-provider", {"provider": "watson"}, _base_config().models)
    assert not ok and "invalid model config" in error


def test_remove_tombstones_yaml_native_and_reset_restores(store: ModelStore) -> None:
    cfg = _base_config()
    ok, error = store.remove_profile("luna", cfg.models)
    assert ok, error
    effective = store.apply(cfg)
    assert "luna" not in effective.models
    assert "default" in effective.models
    ok, _ = store.reset()
    assert "luna" in store.apply(cfg).models


def test_remove_rejects_default_last_and_bound(store: ModelStore) -> None:
    cfg = _base_config()
    ok, error = store.remove_profile("default", cfg.models)
    assert not ok and "cannot be removed" in error
    ok, error = store.remove_profile("luna", cfg.models, agent_bindings={"impl-1": "luna"})
    assert not ok and "rebind first" in error


def test_remove_rejects_unknown_profile(store: ModelStore) -> None:
    ok, error = store.remove_profile("nope", _base_config().models)
    assert not ok and "unknown profile" in error


def test_set_agent_model_overrides_binding(store: ModelStore) -> None:
    cfg = _base_config()
    ok, error = store.set_agent_model("impl-1", "default", cfg.models, cfg.agents)
    assert ok, error
    effective = store.apply(cfg)
    impl = next(a for a in effective.agents if a.agent_id == "impl-1")
    assert impl.model == "default"
    # other agents untouched
    arch = next(a for a in effective.agents if a.agent_id == "arch-1")
    assert arch.model == "default"


def test_set_agent_model_validates(store: ModelStore) -> None:
    cfg = _base_config()
    ok, error = store.set_agent_model("ghost", "default", cfg.models, cfg.agents)
    assert not ok and "unknown agent" in error
    ok, error = store.set_agent_model("impl-1", "ghost-profile", cfg.models, cfg.agents)
    assert not ok and "unknown profile" in error


def test_persistence_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    first = ModelStore(path)
    ok, _ = first.upsert_profile(
        "gpt5", {"provider": "openai-compatible", "name": "gpt-5"}, _base_config().models
    )
    assert ok
    second = ModelStore(path)
    assert "gpt5" in second.apply(_base_config()).models


def test_corrupt_file_degrades_to_yaml(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{not json", encoding="utf-8")
    store = ModelStore(path)
    cfg = _base_config()
    assert store.apply(cfg) is cfg  # no overrides parsed -> identity


def test_non_dict_file_degrades_to_yaml(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON, wrong shape
    store = ModelStore(path)
    cfg = _base_config()
    assert store.apply(cfg) is cfg
    # and it self-heals on the next save
    ok, _ = store.upsert_profile("gpt5", {"provider": "fake"}, cfg.models)
    assert ok
    assert "gpt5" in ModelStore(path).apply(cfg).models


def test_upsert_un_tombstones_a_removed_profile(store: ModelStore) -> None:
    cfg = _base_config()
    ok, _ = store.remove_profile("luna", cfg.models)
    assert ok
    assert "luna" not in store.apply(cfg).models
    # adding it back clears the tombstone
    ok, _ = store.upsert_profile(
        "luna", {"provider": "openai-compatible", "name": "gpt-5.6-luna"}, cfg.models
    )
    assert ok
    assert store.apply(cfg).models["luna"].name == "gpt-5.6-luna"


def test_remove_rejects_last_profile(tmp_path: Path) -> None:
    single = {"only": ModelConfig(provider="fake", name="solo")}
    store = ModelStore(tmp_path / "models.json")
    ok, error = store.remove_profile("only", single)
    assert not ok and "last profile" in error


def test_invalid_stored_profile_is_skipped(tmp_path: Path) -> None:
    import json

    path = tmp_path / "models.json"
    path.write_text(
        json.dumps({"profiles": {"bad": {"temperature": 99}}, "removed": [], "agents": {}}),
        encoding="utf-8",
    )
    store = ModelStore(path)
    effective = store.apply(_base_config())
    assert "bad" not in effective.models
    assert "default" in effective.models


def test_effective_models_with_empty_base(store: ModelStore) -> None:
    assert store.effective_models({}) == {}


def test_profile_cap(store: ModelStore) -> None:
    base = _base_config().models
    for i in range(MAX_PROFILES - len(store.effective_models(base))):
        ok, _ = store.upsert_profile(f"m-{i}", {"provider": "fake"}, base)
        assert ok
    ok, error = store.upsert_profile("one-too-many", {"provider": "fake"}, base)
    assert not ok and "limit" in error


def test_overrides_summary(store: ModelStore) -> None:
    store.upsert_profile("gpt5", {"provider": "fake"}, _base_config().models)
    store.remove_profile("luna", _base_config().models)
    store.set_agent_model("impl-1", "gpt5", _base_config().models, _base_config().agents)
    summary = store.overrides_summary()
    assert summary["profiles"] == ["gpt5"]
    assert summary["removed"] == ["luna"]
    assert summary["agents"] == {"impl-1": "gpt5"}


# -- Service endpoints ----------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    app = create_app(config=_base_config())
    return TestClient(app, raise_server_exceptions=False)


def test_models_listing_shows_effective_and_bindings(client: TestClient) -> None:
    data = client.get("/api/models").json()
    names = [p["profile"] for p in data["profiles"]]
    assert names == ["default", "luna"]
    assert data["agents"]["impl-1"]["model"] == "luna"
    assert data["overrides"]["profiles"] == []
    assert data["default_profile"] == "default"
    # the key VALUE never crosses: only the env-var NAME
    assert "api_key_env" in data["profiles"][1]
    assert data["profiles"][1]["api_key_env"] == "LUNA_API_KEY"


def test_model_crud_roundtrip(client: TestClient) -> None:
    # add
    resp = client.post(
        "/api/models",
        json={
            "profile": "glm",
            "fields": {
                "provider": "openai-compatible",
                "name": "glm-4-plus",
                "base_url": "http://127.0.0.1:8788/v1",
            },
        },
    ).json()
    assert resp["saved"] is True
    names = [p["profile"] for p in client.get("/api/models").json()["profiles"]]
    assert "glm" in names
    glm = next(p for p in client.get("/api/models").json()["profiles"] if p["profile"] == "glm")
    assert glm["managed"] is True and glm["from_yaml"] is False
    # edit
    resp = client.patch(
        "/api/models/glm", json={"profile": "glm", "fields": {"name": "glm-4.5"}}
    ).json()
    assert resp["saved"] is True
    glm = next(p for p in client.get("/api/models").json()["profiles"] if p["profile"] == "glm")
    assert glm["model"] == "glm-4.5"
    # remove
    resp = client.delete("/api/models/glm").json()
    assert resp["saved"] is True
    names = [p["profile"] for p in client.get("/api/models").json()["profiles"]]
    assert "glm" not in names


def test_model_add_rejects_invalid_payload(client: TestClient) -> None:
    resp = client.post("/api/models", json={"profile": "Bad Name", "fields": {}}).json()
    assert resp["saved"] is False
    resp = client.post("/api/models", json={"profile": "hot", "fields": {"temperature": 42}}).json()
    assert resp["saved"] is False


def test_patch_unknown_profile_and_rename_rejected(client: TestClient) -> None:
    resp = client.patch("/api/models/ghost", json={"profile": "ghost", "fields": {}}).json()
    assert resp["saved"] is False and "unknown profile" in resp["error"]
    resp = client.patch("/api/models/luna", json={"profile": "other", "fields": {}}).json()
    assert resp["saved"] is False and "renamed" in resp["error"]


def test_delete_default_rejected(client: TestClient) -> None:
    resp = client.delete("/api/models/default").json()
    assert resp["saved"] is False and "cannot be removed" in resp["error"]


def test_delete_yaml_profile_tombstones_until_reset(client: TestClient) -> None:
    # rebind impl-1 off luna first (bound profiles are protected)
    resp = client.put("/api/agents/impl-1/model", json={"profile": "default"}).json()
    assert resp["saved"] is True
    resp = client.delete("/api/models/luna").json()
    assert resp["saved"] is True
    names = [p["profile"] for p in client.get("/api/models").json()["profiles"]]
    assert "luna" not in names
    assert client.post("/api/models/reset").json()["reset"] is True
    names = [p["profile"] for p in client.get("/api/models").json()["profiles"]]
    assert "luna" in names
    # binding cleared by reset too
    agents = client.get("/api/models").json()["agents"]
    assert agents["impl-1"]["model"] == "luna"


def test_agent_rebind_reflected_in_roster(client: TestClient) -> None:
    resp = client.put("/api/agents/mgr-1/model", json={"profile": "luna"}).json()
    assert resp["saved"] is True
    roster = {a["agent_id"]: a for a in client.get("/api/agents").json()["agents"]}
    assert roster["mgr-1"]["model"] == "luna"
    assert roster["impl-1"]["model"] == "luna"  # yaml binding intact
    assert roster["arch-1"]["model"] == "default"
    # unbind -> follows the run-level picker again
    resp = client.put("/api/agents/mgr-1/model", json={"profile": "default"}).json()
    assert resp["saved"] is True
    roster = {a["agent_id"]: a for a in client.get("/api/agents").json()["agents"]}
    assert roster["mgr-1"]["model"] == "default"


def test_agent_rebind_validates(client: TestClient) -> None:
    resp = client.put("/api/agents/ghost/model", json={"profile": "luna"}).json()
    assert resp["saved"] is False and "unknown agent" in resp["error"]
    resp = client.put("/api/agents/mgr-1/model", json={"profile": "ghost"}).json()
    assert resp["saved"] is False and "unknown profile" in resp["error"]


def test_mutation_reports_pipeline_rebuilds(client: TestClient) -> None:
    resp = client.post("/api/models", json={"profile": "x1", "fields": {"provider": "fake"}}).json()
    assert resp["pipelines_rebuilt"] == 0  # empty cache: nothing to drop


def test_patch_with_invalid_fields_fails_soft(client: TestClient) -> None:
    resp = client.patch(
        "/api/models/luna", json={"profile": "luna", "fields": {"temperature": 99.0}}
    ).json()
    assert resp["saved"] is False and "invalid model config" in resp["error"]


def test_fs_browse_rejects_nul_byte_path(tmp_path: Path) -> None:
    app = create_app(config=_base_config())
    client = TestClient(app)
    data = client.get("/api/fs", params={"path": "bad\x00path"}).json()
    assert data["found"] is False and "invalid path" in data["error"]


def test_fs_browse_skips_dangling_symlink(tmp_path: Path) -> None:
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    (tmp_path / "real.txt").write_text("x")
    app = create_app(config=_base_config())
    client = TestClient(app)
    data = client.get("/api/fs", params={"path": str(tmp_path)}).json()
    assert data["found"] is True
    names = [e["name"] for e in data["entries"]]
    assert "dangling" not in names and "real.txt" in names


def test_fs_browse_truncates_at_500(tmp_path: Path) -> None:
    for i in range(501):
        (tmp_path / f"f{i:03d}").write_text("x")
    app = create_app(config=_base_config())
    client = TestClient(app)
    data = client.get("/api/fs", params={"path": str(tmp_path)}).json()
    assert data["found"] is True
    assert data["truncated"] is True
    assert len(data["entries"]) == 500


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through mode bits")
def test_fs_browse_permission_denied(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        app = create_app(config=_base_config())
        client = TestClient(app)
        data = client.get("/api/fs", params={"path": str(locked)}).json()
        assert data["found"] is False and "permission" in data["error"].lower()
    finally:
        locked.chmod(0o755)


def test_fs_browse_generic_oserror_is_soft(tmp_path: Path, monkeypatch) -> None:
    def boom(self: Path) -> Any:
        raise OSError("disk transient")

    monkeypatch.setattr(Path, "iterdir", boom)
    app = create_app(config=_base_config())
    client = TestClient(app)
    data = client.get("/api/fs", params={"path": str(tmp_path)}).json()
    assert data["found"] is False and "disk transient" in data["error"]


def test_fs_browse_lists_dirs_first(tmp_path: Path) -> None:
    (tmp_path / "zeddir").mkdir()
    (tmp_path / "adir").mkdir()
    (tmp_path / "b.txt").write_text("b")
    app = create_app(config=_base_config())
    client = TestClient(app)
    data = client.get("/api/fs", params={"path": str(tmp_path)}).json()
    assert data["found"] is True
    assert data["path"] == str(tmp_path.resolve())
    assert data["parent"] is not None
    names = [e["name"] for e in data["entries"]]
    assert names.index("adir") < names.index("b.txt")
    assert names.index("zeddir") < names.index("b.txt")
    entry = next(e for e in data["entries"] if e["name"] == "b.txt")
    assert entry["is_dir"] is False and entry["size"] == 1


def test_fs_browse_errors_are_soft(tmp_path: Path) -> None:
    app = create_app(config=_base_config())
    client = TestClient(app)
    missing = tmp_path / "nope"
    data = client.get("/api/fs", params={"path": str(missing)}).json()
    assert data["found"] is False and "not found" in data["error"]
    file_path = tmp_path / "f.txt"
    file_path.write_text("x")
    data = client.get("/api/fs", params={"path": str(file_path)}).json()
    assert data["found"] is False and "not a directory" in data["error"]


def test_fs_browse_root_parent_is_none(tmp_path: Path) -> None:
    app = create_app(config=_base_config())
    client = TestClient(app)
    data = client.get("/api/fs", params={"path": "/"}).json()
    assert data["found"] is True
    assert data["parent"] is None  # "/" is its own parent: stop climbing
