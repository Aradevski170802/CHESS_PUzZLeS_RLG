"""Integration tests for the Flask API in web/backend/app.py.

The module is loaded directly from its file path (rather than `import web.backend.app`)
because `web/backend/` is not necessarily a Python package. Every test that would
otherwise touch real user data redirects `SESSIONS_DIR` to a pytest tmp_path, and the
puzzle pool is replaced with the module's own built-in `DEMO_PUZZLES` fixture data
rather than requiring the real (gitignored) Lichess CSV/Parquet to be present.

Routes that spawn background threads doing live Chess.com/Stockfish calls
(`/api/analysis/start`, `/api/generate/puzzles`, `/api/player/lookup`,
`/api/player/style`) are intentionally out of scope for this offline test module.
"""

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "web" / "backend" / "app.py"


def _seed_verification_challenge(app_module, username, code="TEST-CODE-0000"):
    """Registration now requires proving Chess.com account ownership first
    (POST /api/auth/challenge -> paste code into Chess.com profile ->
    POST /api/auth/verify-chess), both of which make a real network call to
    Chess.com. These tests stay offline by seeding the in-memory challenge
    store directly, exactly as if step 1/2 had already succeeded."""
    app_module._VERIFY_CHALLENGES[username] = {
        "code": code,
        "expires": datetime.now(timezone.utc) + timedelta(seconds=600),
    }
    return code


def _load_app_module():
    spec = importlib.util.spec_from_file_location("chess_puzzle_flask_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    mod = _load_app_module()

    # Never touch the real data/sessions directory from a test run.
    monkeypatch.setattr(mod, "SESSIONS_DIR", tmp_path)

    # auth_register() independently re-checks the Chess.com "location" field
    # against the pending challenge code (a real network call). Fake it to
    # always "contain" whatever code is currently pending for that username,
    # so registration tests stay fully offline.
    monkeypatch.setattr(
        mod,
        "_chess_com_location",
        lambda username: mod._VERIFY_CHALLENGES.get(username, {}).get("code", ""),
    )

    # Use the module's own demo fixture instead of requiring the real puzzle dataset.
    mod.PUZZLE_POOL = list(mod.DEMO_PUZZLES)
    mod._build_category_index()

    # Reset in-memory stores between tests.
    mod.SESSION_STORE.clear()
    mod.ANALYSIS_STORE.clear()
    mod.GENERATE_STORE.clear()

    return mod


@pytest.fixture()
def client(app_module):
    app_module.app.testing = True
    return app_module.app.test_client()


class TestStats:
    def test_stats_reflects_the_loaded_pool(self, client, app_module):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["totalPuzzles"] == len(app_module.DEMO_PUZZLES)
        assert data["categories"] == len(app_module.PUZZLE_BY_CAT)


class TestPuzzleRoutes:
    def test_random_puzzle_within_requested_rating_band(self, client):
        resp = client.get("/api/puzzle/random?ratingMin=1800&ratingMax=2000")
        assert resp.status_code == 200
        data = resp.get_json()
        assert 1800 <= data["rating"] <= 2000

    def test_random_puzzle_no_match_returns_404(self, client):
        resp = client.get("/api/puzzle/random?ratingMin=5000&ratingMax=5000")
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_puzzle_by_known_id(self, client, app_module):
        known_id = app_module.DEMO_PUZZLES[0]["PuzzleId"]
        resp = client.get(f"/api/puzzle/{known_id}")
        assert resp.status_code == 200
        assert resp.get_json()["id"] == known_id

    def test_puzzle_by_unknown_id_returns_404(self, client):
        resp = client.get("/api/puzzle/does-not-exist")
        assert resp.status_code == 404


class TestAdaptiveSession:
    def test_session_start_returns_full_weakness_map(self, client, app_module):
        resp = client.post("/api/session/start", json={"username": "sessiontester"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["returning"] is False
        assert set(data["weaknessMap"].keys()) == set(
            app_module.WEAKNESS_CATEGORIES
        )
        assert len(data["topWeaknesses"]) == 5

    def test_session_puzzle_and_result_round_trip(self, client):
        client.post("/api/session/start", json={"username": "sessiontester"})

        puzzle_resp = client.get("/api/session/puzzle?username=sessiontester")
        assert puzzle_resp.status_code == 200
        puzzle = puzzle_resp.get_json()
        assert "targetCategory" in puzzle
        assert "id" in puzzle

        result_resp = client.post(
            "/api/session/result",
            json={
                "username": "sessiontester",
                "category": puzzle["targetCategory"],
                "solved": True,
                "puzzleId": puzzle["id"],
                "rating": puzzle["rating"],
            },
        )
        assert result_resp.status_code == 200
        result = result_resp.get_json()
        assert result["puzzlesPlayed"] == 1
        assert result["accuracy"] == 100.0

    def test_session_stats_without_active_session_is_404(self, client):
        resp = client.get("/api/session/stats?username=nobody-yet")
        assert resp.status_code == 404


def _play_one(client, username, solved=True):
    puzzle = client.get(f"/api/session/puzzle?username={username}").get_json()
    client.post("/api/session/result", json={
        "username": username, "category": puzzle["targetCategory"], "solved": solved,
        "puzzleId": puzzle["id"], "rating": puzzle["rating"],
    })
    return puzzle


class TestDifficultyAwarePolicy:
    def test_fractional_priors_are_not_truncated(self, client, app_module):
        cat = app_module.WEAKNESS_CATEGORIES[0]
        client.post("/api/session/start", json={"username": "frac", "priors": {cat: [2.6, 7.4]}})
        arm = app_module.SESSION_STORE["frac"].arms[cat]
        assert (arm.alpha, arm.beta) == pytest.approx((2.6, 7.4))

    def test_session_start_reports_policy_and_category_ratings(self, client, app_module):
        data = client.post("/api/session/start", json={"username": "irtuser"}).get_json()
        assert data["policy"] == "irt"
        assert set(data["categoryRatings"]) == set(app_module.WEAKNESS_CATEGORIES)

    def test_puzzle_carries_a_difficulty_aware_prediction(self, client):
        client.post("/api/session/start", json={"username": "irtuser", "estimatedElo": 1500})
        puzzle = client.get("/api/session/puzzle?username=irtuser").get_json()
        assert puzzle["policy"] == "irt"
        assert 0.0 < puzzle["predictedSolveProb"] < 1.0
        assert puzzle["targetSolveRate"] == puzzle["predictedSolveProb"]
        assert puzzle["targetRating"] is not None

    def test_history_logs_policy_propensity_and_prediction(self, client, app_module, tmp_path):
        import json
        client.post("/api/session/start", json={"username": "logger", "estimatedElo": 1400})
        _play_one(client, "logger", solved=False)
        state = json.loads((tmp_path / "logger.json").read_text(encoding="utf-8"))
        entry = state["history"][-1]
        assert entry["policy"] == "irt"
        assert 0.0 <= entry["propensity"] <= 1.0
        assert 0.0 < entry["predicted"] < 1.0
        assert state["irt"]["nUpdates"] == 1

    def test_model_endpoint_tracks_attempts(self, client, app_module):
        client.post("/api/session/start", json={"username": "modeller"})
        for _ in range(3):
            _play_one(client, "modeller")
        data = client.get("/api/session/model/modeller").get_json()
        assert data["attempts"] == 3
        assert len(data["categoryRatings"]) == len(app_module.WEAKNESS_CATEGORIES)

    def test_beta_policy_still_available(self, client, app_module, monkeypatch):
        monkeypatch.setattr(app_module, "RECOMMENDER", "beta")
        client.post("/api/session/start", json={"username": "betauser"})
        puzzle = client.get("/api/session/puzzle?username=betauser").get_json()
        assert puzzle["policy"] == "beta"
        assert puzzle["targetRating"] is None

    def test_returning_player_is_rebuilt_from_history(self, client, app_module, tmp_path):
        import json
        cat = app_module.WEAKNESS_CATEGORIES[0]
        history = [{"ts": "2026-09-01T10:00:00Z", "puzzleId": f"p{i}", "category": cat,
                    "rating": 1400, "solved": i % 2 == 0} for i in range(6)]
        (tmp_path / "veteran.json").write_text(json.dumps({
            "username": "veteran", "history": history, "estimatedElo": 1450,
            "bandit": {"arms": {}, "history": [], "streak": 0, "best_streak": 0},
        }), encoding="utf-8")
        client.post("/api/session/start", json={"username": "veteran"})
        assert app_module.IRT_STORE["veteran"].n_updates == 6


class TestAuth:
    def test_register_then_login(self, client, app_module):
        code = _seed_verification_challenge(app_module, "newuser1")
        reg = client.post(
            "/api/auth/register",
            json={
                "username": "newuser1",
                "password": "secret123",
                "verificationCode": code,
            },
        )
        assert reg.status_code == 200
        assert reg.get_json()["ok"] is True

        login = client.post(
            "/api/auth/login",
            json={"username": "newuser1", "password": "secret123"},
        )
        assert login.status_code == 200
        token = login.get_json()["token"]

        check = client.get(
            "/api/auth/check",
            query_string={"username": "newuser1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert check.status_code == 200
        assert check.get_json()["valid"] is True

    def test_register_rejects_short_password(self, client, app_module):
        code = _seed_verification_challenge(app_module, "shortpw")
        resp = client.post(
            "/api/auth/register",
            json={"username": "shortpw", "password": "abc", "verificationCode": code},
        )
        assert resp.status_code == 400

    def test_register_without_verification_code_is_rejected(self, client):
        # No /api/auth/challenge step completed at all -> missing-field guard.
        resp = client.post(
            "/api/auth/register",
            json={"username": "noverify", "password": "secret123"},
        )
        assert resp.status_code == 400

    def test_register_with_wrong_verification_code_is_rejected(self, client, app_module):
        _seed_verification_challenge(app_module, "badcode", code="RIGHT-CODE")
        resp = client.post(
            "/api/auth/register",
            json={
                "username": "badcode",
                "password": "secret123",
                "verificationCode": "WRONG-CODE",
            },
        )
        assert resp.status_code == 400

    def test_register_duplicate_username_is_rejected(self, client, app_module):
        code = _seed_verification_challenge(app_module, "dupeuser")
        payload = {
            "username": "dupeuser",
            "password": "secret123",
            "verificationCode": code,
        }
        first = client.post("/api/auth/register", json=payload)
        second = client.post("/api/auth/register", json=payload)
        assert first.status_code == 200
        # The account-exists check runs before the (now-consumed) challenge is
        # re-checked, so the second call correctly 409s without needing a fresh code.
        assert second.status_code == 409

    def test_login_with_wrong_password_is_rejected(self, client, app_module):
        code = _seed_verification_challenge(app_module, "wrongpw")
        client.post(
            "/api/auth/register",
            json={
                "username": "wrongpw",
                "password": "correct123",
                "verificationCode": code,
            },
        )
        resp = client.post(
            "/api/auth/login",
            json={"username": "wrongpw", "password": "incorrect123"},
        )
        assert resp.status_code == 401

    def test_auth_check_without_token_is_unauthorized(self, client):
        resp = client.get("/api/auth/check", query_string={"username": "nobody"})
        assert resp.status_code == 401
