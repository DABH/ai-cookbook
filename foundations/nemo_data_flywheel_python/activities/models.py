from pydantic import BaseModel, Field


class CustomizationJobRequest(BaseModel):
    """Request to start a NeMo Customizer fine-tuning job.

    Mirrors the fields of ``POST /v1/customization/jobs``. The ``config``
    string selects a base model + hardware profile registered in your
    NeMo platform (e.g. ``meta/llama-3.2-1b-instruct@v1.0.0+A100``).
    """

    config: str
    dataset_namespace: str = "default"
    dataset_name: str
    output_model: str
    hyperparameters: dict = Field(default_factory=dict)


class JobHandle(BaseModel):
    """Identifier and last-known status for a remote NeMo job."""

    job_id: str
    status: str


class EvaluationJobRequest(BaseModel):
    """Request to start a NeMo Evaluator job against a candidate model.

    ``config`` names an evaluation config (``namespace/name``) registered
    with the Evaluator; the target is built from the customized model.
    """

    namespace: str = "default"
    target_model: str
    config: str


class EvaluationScoreRequest(BaseModel):
    """Where to find the decision metric in the evaluation results.

    ``metric_path`` is a list of keys walked into the results JSON,
    e.g. ``["tasks", "my-task", "metrics", "accuracy", "value"]``.
    """

    job_id: str
    metric_path: list[str]


class DeploymentRequest(BaseModel):
    """Request to deploy the customized model as a NIM deployment."""

    name: str
    namespace: str = "default"
    model: str


class DeploymentResult(BaseModel):
    deployment_id: str
    status: str
