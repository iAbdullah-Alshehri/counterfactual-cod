"""Counterfactual background inpainting backends."""

from __future__ import annotations

import cv2
import numpy as np


class LamaInpainting:
    """CPU inference for the OpenCV Zoo LaMa ONNX checkpoint via ONNX Runtime."""

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install ONNX Runtime: python -m pip install onnxruntime") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, cv2.getNumberOfCPUs()))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=options,
                                            providers=["CPUExecutionProvider"])
        inputs = {item.name for item in self.session.get_inputs()}
        if not {"image", "mask"}.issubset(inputs):
            raise ValueError(f"Unexpected LaMa ONNX inputs: {sorted(inputs)}")
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, image_rgb: np.ndarray, candidate: np.ndarray, *, radius: int = 5,
                 dilation: int = 5) -> np.ndarray:
        binary = (candidate >= 0.5).astype(np.uint8) * 255
        if dilation > 0 and binary.any():
            size = 2 * dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            binary = cv2.dilate(binary, kernel)
        if not binary.any():
            return image_rgb.copy()
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_blob = cv2.dnn.blobFromImage(bgr, 1.0 / 255.0, (512, 512),
                                            (0, 0, 0), swapRB=False, crop=False)
        mask_blob = cv2.dnn.blobFromImage(binary, 1.0, (512, 512),
                                           (0,), swapRB=False, crop=False)
        mask_blob = (mask_blob > 0).astype(np.float32)
        result = self.session.run([self.output_name], {"image": image_blob, "mask": mask_blob})[0]
        result = np.transpose(result[0], (1, 2, 0))
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.resize(result, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def inpaint_opencv(image_rgb: np.ndarray, candidate: np.ndarray, *, radius: int = 5,
                   dilation: int = 5) -> np.ndarray:
    """Fill the candidate region with OpenCV Telea inpainting (classical baseline only)."""
    binary = (candidate >= 0.5).astype(np.uint8) * 255
    if dilation > 0 and binary.any():
        size = 2 * dilation + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary, kernel)
    if not binary.any():
        return image_rgb.copy()
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, binary, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
