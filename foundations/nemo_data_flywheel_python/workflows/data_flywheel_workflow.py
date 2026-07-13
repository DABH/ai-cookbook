from datetime import timedelta

from pydantic import BaseModel
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from activities.models import (
    CustomizationJobRequest,
    DeploymentRequest,
    EvaluationJobRequest,
    EvaluationScoreRequest,
)

# The activities module pulls in httpx, which the workflow sandbox should
# not re-import on every run.
with workflow.unsafe.imports_passed_through():
    from activities.nemo_api import (
        deploy_model,
        get_customization_job,
        get_evaluation_job,
        get_evaluation_score,
        start_customization_job,
        start_evaluation_job,
    )

# Terminal states shared by Customizer and Evaluator jobs.
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "error"}

_START_JOB_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=1),
)


class FlywheelInput(BaseModel):
    """One turn of the data flywheel: fine-tune, evaluate, gate, deploy."""

    customization_config: str
    dataset_name: str
    dataset_namespace: str = "default"
    output_model: str
    eval_config: str
    metric_path: list[str]
    min_score: float = 0.8
    require_approval: bool = False
    deployment_name: str = "flywheel-candidate"
    poll_interval_seconds: int = 30
    job_timeout_hours: int = 12
    approval_timeout_hours: int = 24


class FlywheelResult(BaseModel):
    decision: str  # deployed | rejected_low_score | rejected_by_operator
    customization_job_id: str
    evaluation_job_id: str = ""
    score: float | None = None
    deployment_id: str | None = None


@workflow.defn
class DataFlywheelWorkflow:
    """Durably drives dataset -> fine-tune -> evaluate -> gate -> deploy.

    Every NeMo API call is an activity; the hours-long waits between job
    states are workflow timers. The workflow (and therefore the pipeline)
    survives worker restarts, NeMo hiccups, and GPU queue delays without
    any process needing to stay alive.
    """

    def __init__(self) -> None:
        self._phase = "starting"
        self._approval: bool | None = None

    @workflow.signal
    def approve(self, approved: bool) -> None:
        """Operator decision for gated deployments."""
        self._approval = approved

    @workflow.query
    def current_phase(self) -> str:
        return self._phase

    @workflow.run
    async def run(self, input: FlywheelInput) -> FlywheelResult:
        # 1. Fine-tune: start the Customizer job, then poll durably.
        self._phase = "customizing"
        customization = await workflow.execute_activity(
            start_customization_job,
            CustomizationJobRequest(
                config=input.customization_config,
                dataset_namespace=input.dataset_namespace,
                dataset_name=input.dataset_name,
                output_model=input.output_model,
            ),
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=_START_JOB_RETRY,
        )
        status = await self._poll_job(
            get_customization_job, customization.job_id, input
        )
        if status != "completed":
            raise ApplicationError(
                f"Customization job {customization.job_id} ended {status!r}",
                type="CustomizationFailed",
                non_retryable=True,
            )

        # 2. Evaluate the candidate model.
        self._phase = "evaluating"
        evaluation = await workflow.execute_activity(
            start_evaluation_job,
            EvaluationJobRequest(
                namespace=input.dataset_namespace,
                target_model=input.output_model,
                config=input.eval_config,
            ),
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=_START_JOB_RETRY,
        )
        status = await self._poll_job(get_evaluation_job, evaluation.job_id, input)
        if status != "completed":
            raise ApplicationError(
                f"Evaluation job {evaluation.job_id} ended {status!r}",
                type="EvaluationFailed",
                non_retryable=True,
            )
        score = await workflow.execute_activity(
            get_evaluation_score,
            EvaluationScoreRequest(
                job_id=evaluation.job_id, metric_path=input.metric_path
            ),
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=_START_JOB_RETRY,
        )

        # 3. Quality gate: never promote a model that scores below the bar.
        if score < input.min_score:
            workflow.logger.info(
                "Candidate %s scored %.4f < %.4f; not deploying.",
                input.output_model,
                score,
                input.min_score,
            )
            return FlywheelResult(
                decision="rejected_low_score",
                customization_job_id=customization.job_id,
                evaluation_job_id=evaluation.job_id,
                score=score,
            )

        # 4. Optional human gate: wait durably for an operator signal.
        if input.require_approval:
            self._phase = "awaiting_approval"
            try:
                await workflow.wait_condition(
                    lambda: self._approval is not None,
                    timeout=timedelta(hours=input.approval_timeout_hours),
                )
            except TimeoutError:
                self._approval = False
            if not self._approval:
                return FlywheelResult(
                    decision="rejected_by_operator",
                    customization_job_id=customization.job_id,
                    evaluation_job_id=evaluation.job_id,
                    score=score,
                )

        # 5. Deploy the candidate as a NIM deployment.
        self._phase = "deploying"
        deployment = await workflow.execute_activity(
            deploy_model,
            DeploymentRequest(
                name=input.deployment_name,
                namespace=input.dataset_namespace,
                model=input.output_model,
            ),
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=_START_JOB_RETRY,
        )
        self._phase = "done"
        return FlywheelResult(
            decision="deployed",
            customization_job_id=customization.job_id,
            evaluation_job_id=evaluation.job_id,
            score=score,
            deployment_id=deployment.deployment_id,
        )

    async def _poll_job(self, get_job_activity, job_id: str, input: FlywheelInput) -> str:
        """Poll a remote job until terminal. Waits are durable timers."""
        deadline = workflow.now() + timedelta(hours=input.job_timeout_hours)
        while True:
            handle = await workflow.execute_activity(
                get_job_activity,
                job_id,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=2),
                    maximum_interval=timedelta(minutes=1),
                ),
            )
            if handle.status in _TERMINAL_STATUSES:
                return handle.status
            if workflow.now() >= deadline:
                raise ApplicationError(
                    f"Job {job_id} still {handle.status!r} after "
                    f"{input.job_timeout_hours}h",
                    type="JobTimedOut",
                    non_retryable=True,
                )
            await workflow.sleep(input.poll_interval_seconds)
