import os
from dataclasses import dataclass
from roboflow import Roboflow

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")


@dataclass
class RoboflowUniverseModel(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology, api_key: str, model_id: str, model_version: int = None):
        self.ontology = ontology
        self.api_key = api_key
        self.model_id = model_id
        self.model_version = model_version
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace().project(model_id)
        self.model = self.project.version(model_version).model

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        predictions = self.model.predict(input, confidence = confidence * 100).json()

        # filter out predictions that are not in the ontology
        predictions_data = predictions["predictions"]

        predictions_data = [
            p for p in predictions_data if p["label"] in self.ontology.labels
        ]

        predictions["predictions"] = predictions_data

        detections = sv.Detections.from_roboflow(predictions)

        return detections