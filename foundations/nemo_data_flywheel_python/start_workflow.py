import asyncio
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.data_flywheel_workflow import DataFlywheelWorkflow, FlywheelInput


async def main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    input = FlywheelInput(
        # Base model + hardware profile registered in your NeMo platform.
        customization_config="meta/llama-3.2-1b-instruct@v1.0.0+A100",
        dataset_namespace="default",
        dataset_name="support-tickets-curated",
        output_model="default/llama-3.2-1b-support@v1",
        eval_config="default/support-quality-eval",
        # Where the decision metric lives in your evaluation results JSON.
        metric_path=["tasks", "support-quality", "metrics", "accuracy", "value"],
        min_score=0.8,
        require_approval=False,
    )

    result = await client.execute_workflow(
        DataFlywheelWorkflow.run,
        input,
        id=f"nemo-data-flywheel-{uuid.uuid4()}",
        task_queue="nemo-data-flywheel-task-queue",
    )
    print(f"Decision: {result.decision}")
    print(f"Customization job: {result.customization_job_id}")
    print(f"Evaluation job:    {result.evaluation_job_id}")
    print(f"Score:             {result.score}")
    if result.deployment_id:
        print(f"Deployment:        {result.deployment_id}")


if __name__ == "__main__":
    asyncio.run(main())
