from collections import namedtuple

from loguru import logger
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


class TensorNotFound(Exception):
    pass


TensorInfo = namedtuple("TensorInfo", ["dtype", "shape", "size"])
DeviceBuffer = namedtuple("DeviceBuffer", ["device", "host"])


class TensorRTInference:
    def __init__(self, trt_engine_path: str):
        import pycuda.autoinit

        self.__setup(trt_engine_path)

    def __setup(self, trt_engine_path: str):
        self._trt_engine_path = trt_engine_path
        self._bindings = []
        self._initialise_trt_runtime()
        self._initialise_engine()
        self._initialise_context()
        self._initialise_memory()

    def _initialise_memory(self):
        self._stream = cuda.Stream()
        self._allocate_buffers()

    def _initialise_trt_runtime(self):
        self.logger = trt.Logger(trt.ILogger.ERROR)
        self.runtime = trt.Runtime(self.logger)

    def _initialise_engine(self):
        trt.init_libnvinfer_plugins(None, "")
        with open(self._trt_engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        self._engine = engine

    def _initialise_context(self):
        self._context = self._engine.create_execution_context()

    def _allocate_buffers(self):
        self._allocate_input_buffer()
        self._allocate_output_buffer()

    def _allocate_input_buffer(self):
        input_tensor_name = self._get_input_tensor_name()
        tensor_info = self._get_tensor_information(input_tensor_name)
        host_memory = self._allocate_host_memory(tensor_info)
        device_memory = self._allocate_device_memory(host_memory)
        self._input_buffer = DeviceBuffer(device=device_memory, host=host_memory)
        self._context.set_tensor_address(input_tensor_name, int(device_memory))
        self._bindings.append(int(device_memory))

    def _get_input_tensor_name(self):
        for i in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(i)
            if self._is_input_tensor(tensor_name):
                return tensor_name
        raise TensorNotFound("No Input Tensor Found in TensorRT Engine")

    def _is_input_tensor(self, tensor_name: str):
        tensor_mode = self._engine.get_tensor_mode(tensor_name)
        return tensor_mode == trt.TensorIOMode.INPUT

    def _allocate_output_buffer(self):
        output_tensor_name = self._get_output_tensor_name()
        tensor_info = self._get_tensor_information(output_tensor_name)
        host_memory = self._allocate_host_memory(tensor_info)
        host_memory = host_memory.reshape(tensor_info.shape)
        device_memory = self._allocate_device_memory(host_memory)
        self._output_buffer = DeviceBuffer(device=device_memory, host=host_memory)
        self._context.set_tensor_address(output_tensor_name, int(device_memory))
        self._bindings.append(int(device_memory))

    def _get_output_tensor_name(self):
        for i in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(i)
            if self._is_output_tensor(tensor_name):
                return tensor_name
        raise TensorNotFound("No Output Tensor Found in TensorRT Engine")

    def _is_output_tensor(self, tensor_name: str):
        tensor_mode = self._engine.get_tensor_mode(tensor_name)
        return tensor_mode == trt.TensorIOMode.OUTPUT

    def _get_tensor_information(self, tensor_name: str) -> TensorInfo:
        shape = self._engine.get_tensor_shape(tensor_name)
        size = trt.volume(shape)
        dtype = trt.nptype(self._engine.get_tensor_dtype(tensor_name))
        return TensorInfo(dtype=dtype, shape=shape, size=size)

    @classmethod
    def _allocate_host_memory(cls, tensor_info: TensorInfo) -> np.array:
        return cuda.pagelocked_empty(tensor_info.size, tensor_info.dtype)

    @classmethod
    def _allocate_device_memory(cls, host_memory: np.array) -> cuda.DeviceAllocation:
        return cuda.mem_alloc(host_memory.nbytes)

    def infer(self, input_data: np.array) -> np.array:
        self._copy_input_to_device(input_data)
        # self._context.execute_async_v3(stream_handle=self._stream.handle)
        self._context.execute_v2(self._bindings)
        self._copy_output_to_host()
        return self._output_buffer.host

    def _copy_input_to_device(self, input_data: np.array):
        np.copyto(self._input_buffer.host, input_data.ravel())
        device_buffer = self._input_buffer.device
        # cuda.memcpy_htod_async(device_buffer, self._input_buffer.host, self._stream)
        cuda.memcpy_htod(device_buffer, self._input_buffer.host)

    def _copy_output_to_host(self) -> np.array:
        host_buffer = self._output_buffer.host
        device_buffer = self._output_buffer.device
        # cuda.memcpy_dtoh_async(host_buffer, device_buffer, self._stream)
        # self._stream.synchronize()
        cuda.memcpy_dtoh(host_buffer, device_buffer)
