import os
import cv2

from sahi_batched.models import Yolov8DetectionModel
from sahi_batched import get_sliced_prediction_batched

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_image_local_path, get_image_size

from label_studio_tools.core.utils.params import get_env

import wandb

from pathlib import Path

LABEL_STUDIO_HOST = get_env("HOST")
LABEL_STUDIO_ACCESS_TOKEN = get_env("ACCESS_TOKEN")
LOCAL_FILES_DOCUMENT_ROOT = get_env("LOCAL_FILES_DOCUMENT_ROOT")
WANDB_NAMESPACE = get_env("WANDB_NAMESPACE")
MODEL_DIR = get_env("MODEL_DIR")

wandb.login()
api = wandb.Api()

model_name = "vacocam_model"
model_version = "latest"
artifact = api.artifact(f"{WANDB_NAMESPACE}/{model_name}:{model_version}", type="model")

artifact.download(root=MODEL_DIR)

YOLOV8_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
YOLOV8_MODEL_VERSION = artifact.version

class NewModel(LabelStudioMLBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load model
        self.sahi_model = Yolov8DetectionModel(
            model_path=YOLOV8_MODEL_PATH,
            confidence_threshold=0.3,
            device="cpu",  # or 'cuda:0'
        )

        self.set("model_version", YOLOV8_MODEL_VERSION)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        # print(f'''\
        # Run prediction on {tasks}
        # Length of tasks: {len(tasks)}
        # Received context: {context}
        # Project ID: {self.project_id}
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}''')
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image'
        )

        final_predictions = []
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]

        image_path = get_image_local_path(
            image_url,
            label_studio_host=LABEL_STUDIO_HOST,
            label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
        )

        # Height and width of image
        original_width, original_height = get_image_size(image_path)

        image = cv2.imread(image_path)

        # Creating list for predictions and variable for scores
        predictions = []
        score = 0

        # Getting prediction using model
        results = get_sliced_prediction_batched(
            image,
            self.sahi_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        # boxes = results[0].boxes

        # # Getting mask segments, boxes from model prediction
        for i, result in enumerate(results.object_prediction_list):
            x, y, width, height = result.bbox.to_xywh()

            # Calculating x and y
            x = x / original_width * 100
            y = y / original_height * 100
            width = width / original_width * 100
            height = height / original_height * 100

            # Adding dict to prediction
            predictions.append({
                "from_name" : self.from_name,
                "to_name" : self.to_name,
                "id": str(i),
                "type": "rectanglelabels",
                "score": result.score.value,
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [self.classes[int(result.category.id)]]
                }})

            # Calculating score
            score += result.score.value

        # Append final dicts to final_predictions
        final_predictions.append({
            "result": predictions,
            "model_version": self.get("model_version"),
            "score": score / len(predictions) if len(predictions) > 0 else 0
        })

        return final_predictions

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

