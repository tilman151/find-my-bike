import copy
import logging

import hydra
import label_studio_sdk
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="reset_tasks")
def reset_tasks(config: DictConfig) -> None:
    client = label_studio_sdk.Client(config.host, config.api_key)
    project = client.get_project(config.project_id)
    predictions = [
        _process_task(task, config.prediction_name)
        for task in filter(_bike_filter, _get_labeled_tasks(project))
    ]
    logger.info(f"Create {len(predictions)} predictions")
    project.create_predictions(predictions)


def _process_task(task, prediction_name):
    task_id = task["id"]
    logger.info(f"Process task {task_id}")
    results = _get_cleaned_results(task)
    prediction = {
        "task": task_id,
        "model_version": prediction_name,
        "result": results,
    }

    return prediction


def _get_labeled_tasks(project):
    page_size = 100
    labeled_filter = {
        "conjunction": "and",
        "items": [
            {
                "filter": "filter:tasks:completed_at",
                "operator": "empty",
                "value": False,
                "type": "Datetime",
            }
        ],
    }
    num_annotated_tasks = project.params["total_annotations_number"]
    for page in range(1, num_annotated_tasks // page_size + 2):
        data = project.get_paginated_tasks(
            page=page, filters=labeled_filter, page_size=page_size
        )
        for task in data["tasks"]:
            yield task


def _bike_filter(task) -> bool:
    results = task["annotations"][0]["result"]
    for result in results:
        if result["from_name"] == "bike":
            if result["value"]["choices"][0] == "Bike":
                return True

    return False


def _get_cleaned_results(task):
    results = copy.deepcopy(task["annotations"][0]["result"])
    for result in results:
        if "origin" in result:
            del result["origin"]
        if "score" in result:
            del result["score"]

    return results


if __name__ == "__main__":
    reset_tasks()
