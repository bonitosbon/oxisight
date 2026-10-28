"""Replaceable image classifier; no model weights are bundled."""
import os
import cv2
import numpy as np

class DevelopmentModel:
    mode = 'development'
    version = 'development-no-model'

    def predict(self, image):
        return dict(status='unable_to_determine', confidence=None,
                    message='No trained rust model is configured. Capture and upload are working; rust detection is not available yet.')

class OnnxClassifier:
    mode = 'model'

    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.version = os.getenv('MODEL_VERSION', 'custom-onnx-unvalidated')
        self.size = int(os.getenv('MODEL_INPUT_SIZE', '224'))
        self.threshold = float(os.getenv('MODEL_THRESHOLD', '0.8'))
        self.calibrated = os.getenv('MODEL_CONFIDENCE_CALIBRATED', 'false').lower() == 'true'
        if not 0.5 <= self.threshold <= 1:
            raise ValueError('MODEL_THRESHOLD must be between 0.5 and 1')
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)':
            raise ValueError('Model must have one float32 NCHW input')
        self.input_name = inputs[0].name

    def predict(self, image):
        # Contract: RGB, NCHW float32, ImageNet normalization, two logits:
        # index 0 = no rust; index 1 = rust. Adapt this for your training recipe.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size)).astype(np.float32) / 255.0
        tensor = ((rgb - np.array([.485,.456,.406],dtype=np.float32)) / np.array([.229,.224,.225],dtype=np.float32)).transpose(2,0,1)[None]
        logits = np.asarray(self.session.run(None, {self.input_name:tensor})[0]).reshape(-1)
        if logits.size != 2 or not np.isfinite(logits).all():
            raise ValueError('Expected two finite classification logits')
        scores = np.exp(logits-logits.max()); scores /= scores.sum()
        score = float(scores.max())
        status = ('rust_detected' if scores.argmax() == 1 else 'no_rust_detected') if score >= self.threshold else 'unable_to_determine'
        messages = {'rust_detected':'The model identified signs of rust in this image. This does not measure corrosion depth or structural safety.', 'no_rust_detected':'The model did not identify rust in this image. Hidden corrosion and missed detections are still possible.', 'unable_to_determine':'The model could not reach a reliable decision. Try better lighting and a closer, sharper photo.'}
        return dict(status=status, confidence=score if self.calibrated else None, message=messages[status])

def load_model():
    path = os.getenv('RUST_MODEL_PATH')
    if path:
        return OnnxClassifier(path)
    key = os.getenv('GEMINI_API_KEY')
    if key:
        from backend.gemini import GeminiAssessor
        return GeminiAssessor(key)
    return DevelopmentModel()
