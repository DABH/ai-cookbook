"""Unit tests for the NeMo API activities (httpx mocked)."""

import httpx
import pytest
from temporalio.testing import ActivityEnvironment

from activities.models import EvaluationScoreRequest
from activities.nemo_api import get_evaluation_score


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestGetEvaluationScore:
    @pytest.mark.asyncio
    async def test_extracts_metric_at_path(self, monkeypatch):
        monkeypatch.setenv("NEMO_BASE_URL", "http://nemo.test")
        payload = {
            "tasks": {"support": {"metrics": {"accuracy": {"value": 0.91}}}}
        }

        async def fake_get(self, url, headers=None):
            return _Response(payload)

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        env = ActivityEnvironment()
        score = await env.run(
            get_evaluation_score,
            EvaluationScoreRequest(
                job_id="eval-1",
                metric_path=["tasks", "support", "metrics", "accuracy", "value"],
            ),
        )
        assert score == pytest.approx(0.91)

    @pytest.mark.asyncio
    async def test_missing_metric_path_raises(self, monkeypatch):
        monkeypatch.setenv("NEMO_BASE_URL", "http://nemo.test")

        async def fake_get(self, url, headers=None):
            return _Response({"tasks": {}})

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        env = ActivityEnvironment()
        with pytest.raises(ValueError, match="Metric path"):
            await env.run(
                get_evaluation_score,
                EvaluationScoreRequest(
                    job_id="eval-1", metric_path=["tasks", "missing", "value"]
                ),
            )

    @pytest.mark.asyncio
    async def test_missing_base_url_raises(self, monkeypatch):
        monkeypatch.delenv("NEMO_BASE_URL", raising=False)
        monkeypatch.delenv("NEMO_EVALUATOR_URL", raising=False)
        env = ActivityEnvironment()
        with pytest.raises(RuntimeError, match="NEMO_BASE_URL"):
            await env.run(
                get_evaluation_score,
                EvaluationScoreRequest(job_id="eval-1", metric_path=["a"]),
            )
