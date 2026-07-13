<!--
description: Orchestrate an NVIDIA NeMo microservices data flywheel — fine-tune, evaluate, quality-gate, and deploy — as one durable Temporal workflow.
tags: [foundations, python, nvidia]
priority: 500
-->

# NVIDIA NeMo Data Flywheel with Temporal

A "data flywheel" continuously turns production data into better models: fine-tune a candidate on curated data, evaluate it, and promote it only if it clears a quality bar. On [NVIDIA NeMo microservices](https://docs.nvidia.com/nemo/microservices/latest/), every step is a long-running REST job — customization jobs can sit in a GPU queue for half an hour before they even start, and a full turn of the wheel can take hours.

That makes the pipeline a natural fit for a Temporal workflow:

- **Every wait is a durable timer.** No process has to stay alive while a fine-tuning job trains. Workers can restart, deploy, or crash mid-run; the workflow resumes exactly where it left off.
- **Every NeMo call is a retryable activity.** Transient API failures are retried by Temporal's activity retry policies — the LLM-ops pipeline doesn't need its own retry bookkeeping (this recipe uses no client-side retries).
- **The quality gate is just code.** Score extraction, thresholds, and an optional human approval (a Temporal Signal) live in one place with full history.

```
start_customization_job ──▶ poll (durable) ──▶ start_evaluation_job ──▶ poll (durable)
                                                        │
                                                        ▼
                    deploy_model ◀── [approval signal] ◀── score ≥ min_score?
                                                        │
                                                        ▼ (no)
                                                 rejected_low_score
```

## What this recipe uses

| Step | NeMo microservice | API |
|---|---|---|
| Fine-tune | Customizer | `POST /v1/customization/jobs`, `GET /v1/customization/jobs/{id}` |
| Evaluate | Evaluator | `POST /v1/evaluation/jobs`, `GET /v1/evaluation/jobs/{id}/status`, `…/results` |
| Deploy | Deployment Management | `POST /v1/deployment/model-deployments` |

Endpoint paths can differ between NeMo microservices platform releases — check the [API reference](https://docs.nvidia.com/nemo/microservices/latest/) for your deployment and adjust `activities/nemo_api.py` if needed.

## Prerequisites

- A deployed NeMo microservices platform (Customizer, Evaluator, Deployment Management) with a dataset registered in the Data Store and an evaluation config registered with the Evaluator
- [Temporal CLI](https://docs.temporal.io/cli) for a local dev server
- [`uv`](https://docs.astral.sh/uv/)

Environment variables:

```bash
export NEMO_BASE_URL="https://nemo.your-platform.example.com"   # single ingress, or:
# export NEMO_CUSTOMIZER_URL=... NEMO_EVALUATOR_URL=... NEMO_DEPLOYMENT_URL=...
# export NEMO_API_TOKEN=...    # if your platform requires auth
```

## Run it

Start a local Temporal server:

```bash
temporal server start-dev
```

Install dependencies and start the worker:

```bash
uv sync
uv run python -m worker
```

Edit the model, dataset, and evaluation config names in `start_workflow.py` to match your platform, then kick off a flywheel turn in another terminal:

```bash
uv run python -m start_workflow
```

Watch progress in the Temporal UI (http://localhost:8233) — the workflow's `current_phase` query shows `customizing`, `evaluating`, `awaiting_approval`, or `deploying`.

### Gated deployments

Set `require_approval=True` in `start_workflow.py` and the workflow will pause (durably — for up to `approval_timeout_hours`) after a passing evaluation. Approve or reject from the CLI:

```bash
temporal workflow signal \
  --workflow-id <your-workflow-id> \
  --name approve \
  --input 'true'
```

## Tests

```bash
uv sync
uv run pytest tests/
```

The workflow tests replace the NeMo activities with scripted fakes and run on Temporal's time-skipping test environment, so hours of simulated polling finish in milliseconds. The first run downloads the Temporal test server binary.

## Design notes

- **Polling, not webhooks.** The jobs APIs are poll-based; the workflow polls via a short activity and sleeps on a durable timer between attempts (`poll_interval_seconds`). A `job_timeout_hours` deadline fails the run if a job never reaches a terminal state.
- **Activities stay generic.** Base URLs and auth come from the environment; model names, dataset names, metric paths, and thresholds are workflow inputs — so one worker fleet serves any number of flywheel configurations.
- **Failures are informative.** A failed fine-tune or evaluation fails the workflow with a typed `ApplicationError` (`CustomizationFailed`, `EvaluationFailed`, `JobTimedOut`) rather than deploying a bad candidate; a low score returns a normal `rejected_low_score` result.
