import os
import cv2
from ultralytics import YOLO
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_image_local_path

LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")

class NewModel(LabelStudioMLBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Labels studio ml is broken and doesn't pass the correct label config on the first run
        if self.get("label_config"):
            self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
                self.parsed_label_config, 'RectangleLabels', 'Image'
            )

        # Load model
        self.model = YOLO("vacov8n-vaco-best.pt")

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

        final_predictions = []
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]

        # Getting full URL
        image_path = get_image_local_path(
            image_url,
            label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
            label_studio_host=LABEL_STUDIO_HOST
        )

        image = cv2.imread(image_path)

        # Height and width of image
        original_width, original_height = image.shape[1], image.shape[0]

        # Creating list for predictions and variable for scores
        predictions = []
        score = 0

        # Getting prediction using model
        results = self.model(image)

        boxes = results[0].boxes

        # # Getting mask segments, boxes from model prediction
        for i, box in enumerate(boxes):
            x, y, width, height = box.xywh[0].tolist()

            # Calculating x and y
            x = (x - width / 2) / original_width * 100
            y = (y - height / 2) / original_height * 100
            width = width / original_width * 100
            height = height / original_height * 100

            # Adding dict to prediction
            predictions.append({
                "from_name" : self.from_name,
                "to_name" : self.to_name,
                "id": str(i),
                "type": "rectanglelabels",
                "score": box.conf.item(),
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rectanglelabels": [self.classes[int(box.cls.item())]]
                }})

            # Calculating score
            score += box.conf.item()

        # Append final dicts to final_predictions
        final_predictions.append({
            "result": predictions,
            "model_version": self.get("model_version"),
            "score": score / len(predictions)
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

