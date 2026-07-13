"""Activities that call the NVIDIA NeMo microservices REST APIs.

Each activity is a single, idempotent HTTP call. No client-side retries:
httpx does not retry by default, and Temporal's activity retry policies
(set in the workflow) own all retry behavior.

Endpoints follow the NeMo microservices platform APIs:

- Customizer: ``POST /v1/customization/jobs``, ``GET /v1/customization/jobs/{id}``
- Evaluator:  ``POST /v1/evaluation/jobs``, ``GET /v1/evaluation/jobs/{id}/status``,
  ``GET /v1/evaluation/jobs/{id}/results``
- Deployment: ``POST /v1/deployment/model-deployments``

Paths can shift between platform releases; check the API reference for
your deployment (https://docs.nvidia.com/nemo/microservices/latest/) and
adjust here if yours differ.
"""

import os

import httpx
from temporalio import activity

from activities.models import (
    CustomizationJobRequest,
    DeploymentRequest,
    DeploymentResult,
    EvaluationJobRequest,
    EvaluationScoreRequest,
    JobHandle,
)

_DEFAULT_TIMEOUT = httpx.Timeout(30.0)


def _base_url(service_env: str) -> str:
    """Resolve a service base URL: specific env var, then NEMO_BASE_URL."""
    url = os.environ.get(service_env) or os.environ.get("NEMO_BASE_URL")
    if not url:
        raise RuntimeError(
            f"Set {service_env} or NEMO_BASE_URL to your NeMo platform endpoint."
        )
    return url.rstrip("/")


def _headers() -> dict:
    headers = {"Accept": "application/json"}
    token = os.environ.get("NEMO_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@activity.defn
async def start_customization_job(request: CustomizationJobRequest) -> JobHandle:
    """Create a Customizer fine-tuning job and return its handle."""
    body = {
        "config": request.config,
        "dataset": {
            "namespace": request.dataset_namespace,
            "name": request.dataset_name,
        },
        "output_model": request.output_model,
    }
    if request.hyperparameters:
        body["hyperparameters"] = request.hyperparameters
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.post(
            f"{_base_url('NEMO_CUSTOMIZER_URL')}/v1/customization/jobs",
            json=body,
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()
    return JobHandle(job_id=data["id"], status=data.get("status", "created"))


@activity.defn
async def get_customization_job(job_id: str) -> JobHandle:
    """Fetch the current status of a Customizer job."""
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.get(
            f"{_base_url('NEMO_CUSTOMIZER_URL')}/v1/customization/jobs/{job_id}",
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()
    return JobHandle(job_id=job_id, status=data["status"])


@activity.defn
async def start_evaluation_job(request: EvaluationJobRequest) -> JobHandle:
    """Create an Evaluator job for the candidate model."""
    body = {
        "namespace": request.namespace,
        "target": {"type": "model", "model": request.target_model},
        "config": request.config,
    }
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.post(
            f"{_base_url('NEMO_EVALUATOR_URL')}/v1/evaluation/jobs",
            json=body,
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()
    return JobHandle(job_id=data["id"], status=data.get("status", "created"))


@activity.defn
async def get_evaluation_job(job_id: str) -> JobHandle:
    """Fetch the current status of an Evaluator job."""
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.get(
            f"{_base_url('NEMO_EVALUATOR_URL')}/v1/evaluation/jobs/{job_id}/status",
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()
    status = data.get("status") or data.get("message", "unknown")
    return JobHandle(job_id=job_id, status=status)


@activity.defn
async def get_evaluation_score(request: EvaluationScoreRequest) -> float:
    """Download evaluation results and extract the decision metric."""
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.get(
            f"{_base_url('NEMO_EVALUATOR_URL')}/v1/evaluation/jobs/{request.job_id}/results",
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()
    node = data
    for key in request.metric_path:
        try:
            node = node[key]
        except (KeyError, TypeError) as err:
            raise ValueError(
                f"Metric path {request.metric_path} not found in evaluation "
                f"results for job {request.job_id} (failed at {key!r})."
            ) from err
    return float(node)


@activity.defn
async def deploy_model(request: DeploymentRequest) -> DeploymentResult:
    """Create a NIM deployment for the customized model."""
    body = {
        "name": request.name,
        "namespace": request.namespace,
        "config": {"model": request.model},
    }
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.post(
            f"{_base_url('NEMO_DEPLOYMENT_URL')}/v1/deployment/model-deployments",
            json=body,
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()
    return DeploymentResult(
        deployment_id=data.get("id", request.name),
        status=data.get("status", "created"),
    )
