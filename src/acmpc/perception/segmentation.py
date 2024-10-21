import os
from typing import Dict

from aci.utils.system_monitor import SystemMonitor, track_runtime
from loguru import logger
import numpy as np
import segmentation_models_pytorch as smp
import torch

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")


Segmentation_Monitor = SystemMonitor(300)


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
            vram = torch.cuda.get_device_properties(0).total_memory
            logger.info(
                f"[POD INFO] #CPUS: {os.cpu_count()} "
                f"Device name: {torch.cuda.get_device_name(0)} "
                f"Max memory: {vram / 1e9:.2f}GB"
            )

    def _setup_segmentation_model(self):
        """
        Load model checkpoints
        """
        self._setup_device()
        model = smp.FPN(encoder_name="resnet18", encoder_weights=None, classes=10)
        model.load_state_dict(torch.load(self._model_weights_path, weights_only=True))
        model.eval()
        model.to(self.device)
        if self._compile_model:
            logger.info("Compiling segmentation model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
            dummy_input = torch.randn(
                1,
                3,
                self._height,
                self._width,
                device=self.device,
            )
            model(dummy_input)
        self.model = model

    def _image_to_tensor(self, image: np.array) -> torch.Tensor:
        x = torch.as_tensor(image, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0) / 255.0
        return x.permute(0, 3, 1, 2)

    @track_runtime(Segmentation_Monitor)
    def segment_drivable_area(self, x: np.array) -> np.array:
        x = self._image_to_tensor(x)
        output = self._do_inference(x)
        return self._post_process(output)

    def _do_inference(self, x: torch.tensor) -> torch.tensor:
        with torch.inference_mode():
            output = self.model.predict(x)
        return output

    def _post_process(self, x: torch.Tensor) -> np.array:
        output = torch.argmax(x, dim=1).cpu().numpy().astype(np.uint8)
        vis = np.copy(output)
        output[output > 1] = 0
        return np.squeeze(output), vis
