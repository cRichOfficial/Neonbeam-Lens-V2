from __future__ import annotations

import threading
from concurrent.futures import Future
from functools import partial
from typing import Any

import numpy as np

SHARED_VDEVICE_GROUP_ID = "SHARED"


class _SharedDirectVDevice:
    """Shared exclusive VDevice (picamera2-compatible, no multi-process service)."""

    TARGET: Any = None
    TARGET_REF_COUNT = 0
    _lock = threading.Lock()

    @classmethod
    def acquire(cls) -> Any:
        with cls._lock:
            if cls.TARGET is None:
                from hailo_platform import HailoSchedulingAlgorithm, VDevice

                params = VDevice.create_params()
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                cls.TARGET = VDevice(params)
            cls.TARGET_REF_COUNT += 1
            return cls.TARGET

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            cls.TARGET_REF_COUNT -= 1
            if cls.TARGET_REF_COUNT <= 0 and cls.TARGET is not None:
                cls.TARGET.release()
                cls.TARGET = None
                cls.TARGET_REF_COUNT = 0


class DirectHailoModel:
    """Minimal picamera2-compatible Hailo wrapper using direct VDevice access."""

    def __init__(self, hef_path: str, *, batch_size: int | None = None, output_type: str = "FLOAT32") -> None:
        from hailo_platform import FormatType, HEF

        self.batch_size = batch_size
        self.target = _SharedDirectVDevice.acquire()
        self.hef = HEF(hef_path)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1 if batch_size is None else batch_size)
        self._set_input_output(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_vstream_info()
        self.configured_infer_model = self.infer_model.configure()
        self.num_outputs = len(self.infer_model.outputs)

    def _set_input_output(self, output_type: str) -> None:
        from hailo_platform import FormatType

        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        self.infer_model.input().set_format_type(input_format_type)
        output_format_type = getattr(FormatType, output_type)
        for output in self.infer_model.outputs:
            output.set_format_type(output_format_type)

    def _get_vstream_info(self):
        return self.hef.get_input_vstream_infos(), self.hef.get_output_vstream_infos()

    def get_input_shape(self) -> tuple[int, int, int]:
        return self.input_vstream_info[0].shape

    def callback(self, completion_info, bindings, future, last) -> None:
        if future._has_had_error:
            return
        if completion_info.exception:
            future._has_had_error = True
            future.set_exception(completion_info.exception)
            return

        if self.num_outputs <= 1:
            if self.batch_size is None:
                future._intermediate_result = bindings.output().get_buffer()
            else:
                future._intermediate_result.append(bindings.output().get_buffer())
        else:
            if self.batch_size is None:
                for name in bindings._output_names:
                    future._intermediate_result[name] = bindings.output(name).get_buffer()
            else:
                for name in bindings._output_names:
                    future._intermediate_result[name].append(bindings.output(name).get_buffer())
        if last:
            future.set_result(future._intermediate_result)

    def run_async(self, input_data: np.ndarray) -> Future:
        if self.batch_size is None:
            input_data = np.expand_dims(input_data, axis=0)

        future = Future()
        future._has_had_error = False
        if self.num_outputs <= 1:
            future._intermediate_result = []
        else:
            future._intermediate_result = {output.name: [] for output in self.infer_model.outputs}

        for i, frame in enumerate(input_data):
            last = i == len(input_data) - 1
            bindings = self._create_bindings()
            bindings.input().set_buffer(frame)
            self.configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            self.configured_infer_model.run_async(
                [bindings], partial(self.callback, bindings=bindings, future=future, last=last)
            )
        return future

    def run(self, input_data: np.ndarray) -> Any:
        return self.run_async(input_data).result()

    def _create_bindings(self):
        output_buffers = {
            name: np.empty(self.infer_model.output(name).shape, dtype=np.float32)
            for name in self.infer_model.output_names
        }
        return self.configured_infer_model.create_bindings(output_buffers=output_buffers)

    def close(self) -> None:
        del self.configured_infer_model
        _SharedDirectVDevice.release()
