import os
from typing import Dict

import numpy as np
import torch
from loguru import logger
import segmentation_models_pytorch as smp

from monitor.system_monitor import track_runtime

# V3 drivable FPN
COLOUR_LIST = np.array(
    [
        (0, 0, 0),
        (0, 255, 249),
        (84, 84, 84),
        (255, 119, 51),
        (255, 255, 255),
        (255, 255, 0),
        (170, 255, 128),
        (255, 42, 0),
        (153, 153, 255),
        (255, 179, 204),
    ]
)


class TrackSegmenter:
    def __init__(self, cfg: Dict):
        self.__setup_config(cfg)
    
    def __setup_config(self, cfg: Dict):
        self._model_weights_path = cfg["model_path"]
        self._width = cfg["image_width"]
        self._height = cfg["image_height"]
        self._compile_model = cfg["compile_model"]

    def _setup_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            logger.info(f"[POD INFO] #CPUS: {os.cpu_count()}")
            logger.error("[RUNTIME ERROR] Experiment not running on a gpu")
            raise Exception("MODEL MUST BE ON GPU, ABORT")
        else:
            logger.info(
                f"[POD INFO] #CPUS: {os.cpu_count()} "
                f"Device name: {torch.cuda.get_device_name(0)} "
                f"Max memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB"
            )

    def _setup_segmentation_model(self):
        """
        Load model checkpoints
        """
        self._setup_device()
        model = smp.FPN(encoder_name="resnet18", encoder_weights=None, classes=10)
        model.load_state_dict(torch.load(self._model_weights_path))
        model.eval()
        model.to(self.device)
        if self._compile_model:
            logger.info("Compiling segmentation model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
            dummy_input = torch.randn(1, 3, self._height, self._width, device=self.device)
            model(dummy_input)
        self.model = model

    def _image_to_tensor(self, image: np.array) -> torch.Tensor:
        x = np.stack([image]) / 255
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return x.permute(0, 3, 1, 2)

    def segment_drivable_area(self, x: np.array):
        x = self._image_to_tensor(x)
        output = self.model.predict(x)
        output = torch.argmax(output, dim=1).cpu().numpy().astype(np.uint8)
        # TODO: Move colour conversion to visualisation process
        vis = np.squeeze(np.array(COLOUR_LIST[output], dtype=np.uint8))
        output[output > 1] = 0
        return np.squeeze(output), vis
