import os
from typing import Dict

import numpy as np
import torch
from loguru import logger
import segmentation_models_pytorch as smp


from models.efficientnet_v2 import EfficientNetV2_FPN_Segmentation
from models.deeplab import resnet, deeplabv3plus
from monitor.system_monitor import track_runtime

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


class TrackSegmenter:
    def __init__(self, cfg: Dict):
        self.model = self.load_segmentation_model(cfg["model_path"])

    def load_segmentation_model(self, path):
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
        # model = EfficientNetV2_FPN_Segmentation(
        #    version="efficientnet_v2_s", im_c=3, n_classes=2
        # ).to(self.device)

        encoder = resnet.ResnetEncoder(resnet.build("18", False), 16)
        model = deeplabv3plus.DeepLabV3plus(encoder, 10).to(self.device)
        # model = smp.FPN(encoder_name="resnet18", weights=None, classes=10)
        modified_state_dict = {}
        state_dict = torch.load(path)["state_dict"]
        for key in state_dict.keys():
            modified_state_dict[key.replace("_model.", "")] = state_dict[key]
        model.load_state_dict(modified_state_dict)
        model.eval()
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
        output = self.model.predict(x)
        output = torch.argmax(output, dim=1).cpu().numpy().astype(np.uint8)
        vis = np.squeeze(np.array(COLOUR_LIST[output], dtype=np.uint8))
        # output[output != 0] = 1
        logger.info(f"Before: {np.unique(output)}")
        output[output == 0] = 2
        output -= 1
        output[output != 1] = 0
        logger.info(f"After: {np.unique(output)}")
        return output, vis
