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
