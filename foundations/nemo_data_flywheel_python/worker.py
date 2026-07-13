import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from activities.nemo_api import (
    deploy_model,
    get_customization_job,
    get_evaluation_job,
    get_evaluation_score,
    start_customization_job,
    start_evaluation_job,
)
from workflows.data_flywheel_workflow import DataFlywheelWorkflow


async def worker_main():
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue="nemo-data-flywheel-task-queue",
        workflows=[DataFlywheelWorkflow],
        activities=[
            start_customization_job,
            get_customization_job,
            start_evaluation_job,
            get_evaluation_job,
            get_evaluation_score,
            deploy_model,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(worker_main())
