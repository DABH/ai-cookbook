"""Workflow tests for DataFlywheelWorkflow with scripted NeMo activities.

The real activities are replaced by FakeNemo stubs registered under the
same activity names, so the tests exercise the full durable pipeline
(polling loops, quality gate, approval signal) with no NeMo platform.
Runs on the time-skipping test environment; the first run downloads the
Temporal test server binary.
"""

import uuid

import pytest
from temporalio import activity
from temporalio.client import Client, WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.models import (
    CustomizationJobRequest,
    DeploymentRequest,
    DeploymentResult,
    EvaluationJobRequest,
    EvaluationScoreRequest,
    JobHandle,
)
from workflows.data_flywheel_workflow import DataFlywheelWorkflow, FlywheelInput


class FakeNemo:
    """Scripted stand-in for the NeMo microservice activities."""

    def __init__(
        self,
        customization_statuses=("running", "completed"),
        evaluation_statuses=("running", "completed"),
        score=0.9,
    ):
        self._customization = list(customization_statuses)
        self._evaluation = list(evaluation_statuses)
        self._score = score
        self.deploy_calls: list[DeploymentRequest] = []

    def _advance(self, statuses: list) -> str:
        return statuses.pop(0) if len(statuses) > 1 else statuses[0]

    @activity.defn(name="start_customization_job")
    async def start_customization_job(
        self, request: CustomizationJobRequest
    ) -> JobHandle:
        return JobHandle(job_id="cust-123", status="created")

    @activity.defn(name="get_customization_job")
    async def get_customization_job(self, job_id: str) -> JobHandle:
        return JobHandle(job_id=job_id, status=self._advance(self._customization))

    @activity.defn(name="start_evaluation_job")
    async def start_evaluation_job(self, request: EvaluationJobRequest) -> JobHandle:
        return JobHandle(job_id="eval-456", status="created")

    @activity.defn(name="get_evaluation_job")
    async def get_evaluation_job(self, job_id: str) -> JobHandle:
        return JobHandle(job_id=job_id, status=self._advance(self._evaluation))

    @activity.defn(name="get_evaluation_score")
    async def get_evaluation_score(self, request: EvaluationScoreRequest) -> float:
        return self._score

    @activity.defn(name="deploy_model")
    async def deploy_model(self, request: DeploymentRequest) -> DeploymentResult:
        self.deploy_calls.append(request)
        return DeploymentResult(deployment_id="dep-1", status="created")


def _flywheel_input(**overrides) -> FlywheelInput:
    defaults = dict(
        customization_config="meta/llama-3.2-1b-instruct@v1.0.0+A100",
        dataset_name="support-tickets-curated",
        output_model="default/llama-3.2-1b-support@v1",
        eval_config="default/support-quality-eval",
        metric_path=["tasks", "support-quality", "metrics", "accuracy", "value"],
        min_score=0.8,
    )
    defaults.update(overrides)
    return FlywheelInput(**defaults)


async def _client_with_pydantic(env: WorkflowEnvironment) -> Client:
    config = env.client.config()
    config["data_converter"] = pydantic_data_converter
    return Client(**config)


def _worker(client: Client, fake: FakeNemo) -> Worker:
    return Worker(
        client,
        task_queue="flywheel-test",
        workflows=[DataFlywheelWorkflow],
        activities=[
            fake.start_customization_job,
            fake.get_customization_job,
            fake.start_evaluation_job,
            fake.get_evaluation_job,
            fake.get_evaluation_score,
            fake.deploy_model,
        ],
    )


class TestDataFlywheelWorkflow:
    @pytest.mark.asyncio
    async def test_deploys_when_score_clears_threshold(self):
        fake = FakeNemo(score=0.92)
        async with await WorkflowEnvironment.start_time_skipping() as env:
            client = await _client_with_pydantic(env)
            async with _worker(client, fake):
                result = await client.execute_workflow(
                    DataFlywheelWorkflow.run,
                    _flywheel_input(),
                    id=f"flywheel-{uuid.uuid4()}",
                    task_queue="flywheel-test",
                )
        assert result.decision == "deployed"
        assert result.score == pytest.approx(0.92)
        assert result.deployment_id == "dep-1"
        assert len(fake.deploy_calls) == 1
        assert fake.deploy_calls[0].model == "default/llama-3.2-1b-support@v1"

    @pytest.mark.asyncio
    async def test_low_score_is_not_deployed(self):
        fake = FakeNemo(score=0.55)
        async with await WorkflowEnvironment.start_time_skipping() as env:
            client = await _client_with_pydantic(env)
            async with _worker(client, fake):
                result = await client.execute_workflow(
                    DataFlywheelWorkflow.run,
                    _flywheel_input(),
                    id=f"flywheel-{uuid.uuid4()}",
                    task_queue="flywheel-test",
                )
        assert result.decision == "rejected_low_score"
        assert result.score == pytest.approx(0.55)
        assert result.deployment_id is None
        assert fake.deploy_calls == []

    @pytest.mark.asyncio
    async def test_operator_can_reject_gated_deployment(self):
        fake = FakeNemo(score=0.95)
        async with await WorkflowEnvironment.start_time_skipping() as env:
            client = await _client_with_pydantic(env)
            async with _worker(client, fake):
                handle = await client.start_workflow(
                    DataFlywheelWorkflow.run,
                    _flywheel_input(require_approval=True),
                    id=f"flywheel-{uuid.uuid4()}",
                    task_queue="flywheel-test",
                )
                await handle.signal(DataFlywheelWorkflow.approve, False)
                result = await handle.result()
        assert result.decision == "rejected_by_operator"
        assert fake.deploy_calls == []

    @pytest.mark.asyncio
    async def test_failed_customization_fails_the_run(self):
        fake = FakeNemo(customization_statuses=("running", "failed"))
        async with await WorkflowEnvironment.start_time_skipping() as env:
            client = await _client_with_pydantic(env)
            async with _worker(client, fake):
                with pytest.raises(WorkflowFailureError) as exc_info:
                    await client.execute_workflow(
                        DataFlywheelWorkflow.run,
                        _flywheel_input(),
                        id=f"flywheel-{uuid.uuid4()}",
                        task_queue="flywheel-test",
                    )
        cause = exc_info.value.cause
        assert isinstance(cause, ApplicationError)
        assert cause.type == "CustomizationFailed"
        assert fake.deploy_calls == []
