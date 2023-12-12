import cv2
from dataclasses import dataclass
from typing import List, Tuple
from roboflow import Roboflow
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

@dataclass
class RoboflowUniverseModel(DetectionBaseModel):
    ontology: CaptionOntology
    model_configs: List[Tuple[str, int]] 

    def __init__(self, ontology: CaptionOntology, api_key: str, model_configs: List[Tuple[str, int]]):
        self.ontology = ontology
        self.api_key = api_key
        self.model_configs = model_configs
        self.models = []
        self.rf = Roboflow(api_key=api_key)

        for model_id, model_version in model_configs:
            project = self.rf.workspace().project(model_id)
            model = project.version(model_version).model
            self.models.append(model)
            
    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        all_predictions = []
        class_name_to_id = {name: id for id, name in enumerate(self.ontology.prompts())}
        image = cv2.imread(input)

        if image is None:
            raise ValueError(f"Unable to read image: {input}")
        image_info = {"width": image.shape[1], "height": image.shape[0]}

        for model in self.models:
            result = model.predict(input, confidence=confidence * 100).json()
            predictions = result["predictions"]

            for pred in predictions:
                pred_class_id = class_name_to_id.get(pred["class"], -1)
                if pred_class_id != -1:
                    pred["class_id"] = pred_class_id
                    all_predictions.append(pred)

        detections = sv.Detections.from_roboflow({"predictions": all_predictions, "image": image_info})
        return detections
