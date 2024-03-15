import os
import time
from typing import Dict

import numpy as np
import torch
from loguru import logger
import segmentation_models_pytorch as smp

from monitor.system_monitor import track_runtime

# V2 FPN
COLOUR_LIST = np.array(
    [
        (84, 84, 84),
        (255, 119, 51),
        (255, 255, 255),
        (255, 255, 0),
        (170, 255, 128),
        (255, 42, 0),
        (153, 153, 255),
        (0, 255, 238),
        (255, 179, 204),
        (0, 102, 17),
        (0, 0, 255),
        (0, 0, 0),
    ]
)
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
        self.model = self.load_segmentation_model(cfg)

    def load_segmentation_model(self, cfg: Dict):
        """
        Load model checkpoints.
        """
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
        model = smp.FPN(encoder_name="resnet18", encoder_weights=None, classes=10)
        model.load_state_dict(torch.load(cfg["model_path"]))
        model.eval()
        model.to(self.device)
        #model = torch.compile(model, mode="reduce-overhead")
        #dummy_input = torch.randn(1, 3, cfg["image_height"], cfg["image_width"], device=self.device)
        #model(dummy_input)
        return model

    def add_inferred_segmentation_masks(self, obs: Dict):
        images = obs.get_images()
        image_tensor = self.images_to_tensor(images)
        masks, vis = self.segment_drivable_area(image_tensor)
        obs.add_segmentation_masks(masks)
        obs["vis"] = vis

    def images_to_tensor(self, images: np.array) -> torch.Tensor:
        x = np.stack(images) / 255
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return x.permute(0, 3, 1, 2)

    @track_runtime
    def segment_drivable_area(self, x: torch.Tensor) -> np.array:
        before = time.time()
        output = self.model.predict(x)
        output = torch.argmax(output, dim=1).cpu().numpy().astype(np.uint8)
        vis = np.squeeze(np.array(COLOUR_LIST[output], dtype=np.uint8))
        output[output > 1] = 0
        return output, vis
