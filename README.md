# Photo Scanner Project

## Setup

This project uses Poetry for dependency management.

### Prerequisites

1.  Python 3.12+
2.  Poetry

### Installation

```bash
poetry install
```

## Model Setup

The MobileCLIP TFLite model is located in `assets/mobileclip_s1_datacompdr_first.tflite`.

Dependencies installed:
-   `tensorflow` (for TFLite execution)
-   `numpy`
-   `pillow`

## Usage

### Inspecting the Model
To see the input/output signatures of the model:

```bash
poetry run python inspect_model.py
```

### Running Inference (Python)
See `src/model_handler.py` for the implementation of the `MobileCLIPHandler` class.

```python
from src.model_handler import MobileCLIPHandler
import os

handler = MobileCLIPHandler(os.path.join("assets", "mobileclip_s1_datacompdr_first.tflite"))

# Get embedding for an image
# Returns tuple (output_0, output_1) - likely (image_embedding, text_embedding)
emb_0, emb_1 = handler.get_image_embedding("path/to/image.jpg")
print(emb_0.shape)
```
