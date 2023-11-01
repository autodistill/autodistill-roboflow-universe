<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Roboflow Universe Module

This repository contains the code supporting the Roboflow Universe base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Roboflow Universe](https://universe.roboflow.com) is a community where people share computer vision models and datasets. Over 50,000 models and 250,000 datasets have been shared on Universe, with new models available every day. You can use Autodistill to run object detection, classification, and segmentation models hosted on Roboflow Universe.

> [!NOTE]
> Using this project will use Roboflow API calls. You will need a free Roboflow account to use this project. [Sign up for a free Roboflow account](https://app.roboflow.com) to get started. [Learn more about pricing](https://roboflow.com/pricing).

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [Roboflow Universe Autodistill documentation](https://autodistill.github.io/autodistill/base_models/roboflow_universe/).

## Installation

To use models hosted on Roboflow Universe with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-roboflow-universe
```

## Quickstart

```python
from autodistill_roboflow_universe import RoboflowUniverseModel
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our Roboflow model prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = RoboflowUniverseModel(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    ),
    model_id="MODEL_ID",
    api_key="API_KEY",
    model_version=VERSION,
    model_type="object-detection",
)

# predict on an image
predictions = base_model.predict("image.png")

print(predictions)

# label a folder of images
base_model.label("./context_images", extension=".jpeg")
```

Above, replace:

- `API_KEY`: with your Roboflow API key
- `PROJECT_NAME`: with your Roboflow project ID.
- `VERSION`: with your Roboflow model version.
- `model_type`: with the type of model you want to run. Options are `object-detection`, `classification`, or `segmentation`. This value must be the same as the model type trained on Roboflow Universe.

[Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).
[Learn how to retrieve a model ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids).

## License

This project is licensed under an [MIT license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!