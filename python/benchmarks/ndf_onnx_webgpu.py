"""Experimental fixed-shape ONNX/WebGPU backend for the learned pair matcher.

The WebGPU execution provider owns its own device. Arrays still cross the host
boundary; this adapter does not claim buffer sharing with wgpu-py or residency.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ndf_pair_model import PairPrediction


class _PairExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, reference, target):
        prediction = self.model(reference, target, include_topk=False)
        return (prediction.displacement_xy, prediction.target_xy, prediction.hard_displacement_xy,
                prediction.hard_target_xy, prediction.confidence)


def _register_webgpu():
    import onnxruntime as ort
    import onnxruntime_ep_webgpu as ep

    registered = [device for device in ort.get_ep_devices() if device.ep_name == ep.get_ep_name()]
    if not registered:
        ort.register_execution_provider_library("ndf_webgpu_ep", ep.get_library_path())
        registered = [device for device in ort.get_ep_devices() if device.ep_name == ep.get_ep_name()]
    devices = [device for device in registered if device.device.type == ort.OrtHardwareDeviceType.GPU]
    if not devices:
        raise RuntimeError("ONNX Runtime WebGPU provider loaded but no device was discovered")
    return ort, devices


class NDFOnnxWebGPU:
    """Run both the encoder and whole pair graph via ONNX Runtime WebGPU."""

    def __init__(self, model, weights, height, width, directory, *, profiling=False):
        self.height, self.width = int(height), int(width)
        if self.height < 4 or self.width < 4:
            raise ValueError("model dimensions must each be at least four")
        cells = ((self.height + 3) // 4) * ((self.width + 3) // 4)
        if cells > 4096:
            raise ValueError("coarse correlation exceeds the 4096-cell bounded export limit")
        model_path = Path(__file__).with_name("ndf_pair_model.py")
        weight_bytes = Path(weights).read_bytes()
        self.weights_sha256 = hashlib.sha256(weight_bytes).hexdigest()
        digest = hashlib.sha256(weight_bytes + model_path.read_bytes()
                                + Path(__file__).read_bytes() + torch.__version__.encode()).hexdigest()
        self.model_id = digest
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.paths = {name: directory / f"ndf_{name}_{self.width}x{self.height}_{digest[:16]}.onnx"
                      for name in ("encoder", "pair")}
        example = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32)
        for name, module, inputs, input_names, output_names in (
            ("encoder", model.encoder, (example,), ["image"], ["features"]),
            ("pair", _PairExport(model), (example, example), ["reference", "target"],
             ["displacement", "target_xy", "hard_displacement", "hard_target", "confidence"]),
        ):
            if not self.paths[name].is_file():
                tmp = self.paths[name].with_suffix(".tmp.onnx")
                try:
                    torch.onnx.export(module, inputs, str(tmp), input_names=input_names,
                                      output_names=output_names, opset_version=17, dynamo=False)
                    tmp.replace(self.paths[name])
                finally:
                    tmp.unlink(missing_ok=True)
        ort, devices = _register_webgpu()
        self.device_info = [{"type": str(device.device.type), "vendor": device.device.vendor,
                             "vendor_id": device.device.vendor_id, "provider": device.ep_name,
                             "provider_metadata": dict(device.ep_metadata)} for device in devices]
        self.sessions = {}
        for name, path in self.paths.items():
            options = ort.SessionOptions()
            options.intra_op_num_threads = min(4, torch.get_num_threads())
            options.inter_op_num_threads = 1
            options.enable_profiling = profiling
            if profiling:
                options.profile_file_prefix = str(directory / f"profile_{name}")
            options.add_provider_for_devices(devices, {"preferredLayout": "NHWC"})
            self.sessions[name] = ort.InferenceSession(str(path), sess_options=options)
            if "WebGpuExecutionProvider" not in self.sessions[name].get_providers():
                raise RuntimeError("WebGPU provider was not selected")
        self.log_temperature = model.log_temperature.detach()

    def _input(self, image):
        if tuple(image.shape) != (1, 1, self.height, self.width) or image.device.type != "cpu":
            raise ValueError("ONNX WebGPU export requires its fixed image shape and a CPU tensor")
        return np.ascontiguousarray(image.detach().numpy(), dtype=np.float32)

    def encoder(self, image):
        result = self.sessions["encoder"].run(None, {"image": self._input(image)})[0]
        return torch.from_numpy(result)

    def __call__(self, reference, target, *, include_topk=False):
        if include_topk:
            raise ValueError("ONNX WebGPU path currently supports soft and hard matching, not topk8")
        result = self.sessions["pair"].run(None, {"reference": self._input(reference), "target": self._input(target)})
        flow, target_xy, hard, hard_target, confidence = (torch.from_numpy(value) for value in result)
        return PairPrediction(flow, target_xy, hard, hard_target, None, None, confidence)

    def finish_profiles(self):
        return {name: session.end_profiling() for name, session in self.sessions.items()}

    def metadata(self):
        return {"backend": "onnxruntime-webgpu", "checkpoint_sha256": self.weights_sha256,
                "export_fingerprint_sha256": self.model_id,
                "shape": [1, 1, self.height, self.width],
                "graphs": {name: str(path) for name, path in self.paths.items()},
                "providers": {name: session.get_providers() for name, session in self.sessions.items()},
                "devices": self.device_info,
                "intermediate_buffers_shared_with_wgpu_py": False}

    def write_metadata(self, path):
        Path(path).write_text(json.dumps(self.metadata(), indent=2) + "\n")
