import ctypes
import multiprocessing as mp
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

class TrackSegmentorProcess(mp.Process):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.__setup(cfg)

    def __setup(self, cfg: Dict):
        self.__setup_config(cfg)
        self.__setup_shared_memory()
    
    def __setup_config(self, cfg: Dict):
        self._model_weights_path = cfg["model_path"]
        self._width = cfg["image_width"]
        self._height = cfg["image_height"]
        self._compile_model = cfg["compile_model"]


    def __setup_device(self):
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

    def __setup_shared_memory(self):
        self._shared_input_buffer = SharedImage(self._height, self._width, 3)
        self._shared_mask_buffer = SharedImage(1, self._height, self._width)
        self._shared_visualisation_buffer = SharedImage(self._height, self._width, 3)
        self._is_running = mp.Value("i", True)

    @property
    def is_running(self) -> bool:
        """
        Checks if the segmentation process is running

        :return: True if the segmentation process is running, false if it is not
        :rtype: bool
        """
        with self._is_running.get_lock():
            is_running = self._is_running.value
        return is_running

    @is_running.setter
    def is_running(self, is_running: bool):
        """
        Sets if the segmentation process is running

        :is_running: True if the segmentation process is running, false if it is not
        :type is_running: bool
        """
        with self._is_running.get_lock():
            self._is_running.value = is_running
    
    @property
    def input_image(self) -> np.array:
        return self._shared_input_buffer.image
    
    @input_image.setter
    def input_image(self, image: np.array):
        self._shared_input_buffer.image = image

    @property
    def output_mask(self) -> np.array:
        return self._shared_mask_buffer.image
    
    @property
    def output_visualisation(self) -> np.array:
        return self._shared_visualisation_buffer.image
    
    @property
    def is_mask_stale(self) -> bool:
        return self._shared_mask_buffer.is_stale
    
    def run(self):
        """
        Called on TrackSegmentorProcess.start()
        """
        self.__setup_segmentation_model()
        while self.is_running:
            image = self._shared_input_buffer.fresh_image
            image_tensor = self._image_to_tensor(image)
            self._segment_drivable_area(image_tensor)
    
    def __setup_segmentation_model(self):
        """
        Load model checkpoints
        """
        self.__setup_device()
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

    def _segment_drivable_area(self, x: torch.Tensor) -> np.array:
        output = self.model.predict(x)
        output = torch.argmax(output, dim=1).cpu().numpy().astype(np.uint8)
        vis = np.squeeze(np.array(COLOUR_LIST[output], dtype=np.uint8))
        output[output > 1] = 0
        self._shared_mask_buffer.image = output
        self._shared_visualisation_buffer.image = vis


class SharedImage:
    def __init__(self, height: int, width: int, channels: int):
        self._image_shape = (height, width, channels)
        self.__setup()
    
    def __setup(self):
        mp_array = mp.Array(ctypes.c_uint8, self._n_pixels)
        np_array = np.ndarray(
            self._image_shape, dtype=np.uint8, buffer=mp_array.get_obj()
        )
        self._shared_image_buffer = (mp_array, np_array)
        self._is_stale = mp.Value("i", True)
    
    @property
    def _n_pixels(self):
        return int(np.prod(self._image_shape))
    
    @property
    def image(self) -> np.array:
        image_mp_array, image_np_array = self._shared_image_buffer
        with image_mp_array.get_lock():
            image = image_np_array.copy()
        self.is_stale = True
        return image
    
    @image.setter
    def image(self, image: np.array):
        image_mp_array, image_np_array = self._shared_image_buffer
        with image_mp_array.get_lock():
            image_np_array[:] = image
        self.is_stale = False
    
    @property
    def fresh_image(self) -> np.array:
        self._wait_for_fresh_image()
        return self.image
        
    def _wait_for_fresh_image(self):
        while self.is_stale:
            continue

    @property
    def is_stale(self) -> bool:
        """
        Checks if the current image has been read by any consumer

        :return: True if the image has been read, false if it has not
        :rtype: bool
        """
        with self._is_stale.get_lock():
            is_stale = self._is_stale.value
        return is_stale


    @is_stale.setter
    def is_stale(self, is_stale: bool):
        """
        Sets the flag indicating if the image has been read previously

        :is_stale: True if the image has been read, false if it has not
        :type is_stale: bool
        """
        with self._is_stale.get_lock():
            self._is_stale.value = is_stale