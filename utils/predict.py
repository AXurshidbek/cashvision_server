from ultralytics import YOLO

MODEL_PATH = "model/cash_model.pt"

model = YOLO(MODEL_PATH)

names = model.names if hasattr(model, "names") else {}
print("Class names:", names)

def predict_image(image_path: str) -> str:

    results = model(image_path, imgsz=640)
    if len(results) == 0:
        return "unknown"
    r = results[0]

    try:
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:

            cls_tensor = getattr(boxes, "cls", None)
            conf_tensor = getattr(boxes, "conf", None)
            if cls_tensor is not None and conf_tensor is not None:
                try:
                    best_idx = int(conf_tensor.argmax().item())
                except Exception:
                    best_idx = 0
                try:
                    cls_idx = int(cls_tensor[best_idx].item())
                except Exception:
                    cls_idx = int(cls_tensor[best_idx])

                label = names.get(cls_idx, str(cls_idx))
                return str(label)
    except Exception as e:

        print("Debug: detection parsing error:", e)

    try:
        probs = getattr(r, "probs", None)
        if probs is not None:
            if hasattr(probs, "top1"):
                try:
                    top1 = int(probs.top1.item())
                except Exception:
                    top1 = int(probs.top1)
            else:
                try:
                    top1 = int(probs.argmax().item())
                except Exception:
                    top1 = int(probs.argmax())
            label = names.get(top1, str(top1))
            return str(label)
    except Exception as e:
        print("Debug: classification parsing error:", e)

    return "unknown"

#=========================================================================

#
# import onnxruntime as ort
# import numpy as np
# import cv2
#
# MODEL_PATH = "model/cash_model.onnx"
#
# session = ort.InferenceSession(MODEL_PATH)
#
# CLASS_NAMES = {0: '2000', 1: '5000', 2: '10000', 3: '20000', 4: '50000', 5: '100000'}
#
# CONF_THRESHOLD = 0.5
#
#
# def preprocess(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (640, 640))
#     img = img.astype(np.float32) / 255.0
#     img = np.transpose(img, (2, 0, 1))
#     img = np.expand_dims(img, axis=0)
#     return img
#
#
# def predict_image(image_path: str) -> str:
#     img = preprocess(image_path)
#
#     outputs = session.run(None, {"images": img})
#     detections = outputs[0][0]  # (300, 6)
#
#     best_conf = 0
#     best_class = None
#
#     for det in detections:
#         x1, y1, x2, y2, conf, class_id = det
#
#         if conf > CONF_THRESHOLD and conf > best_conf:
#             best_conf = conf
#             best_class = int(class_id)
#
#     if best_class is not None:
#         return CLASS_NAMES.get(best_class, "unknown")
#
#     return "unknown"