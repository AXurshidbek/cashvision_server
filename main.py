from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
import uuid, os

from utils.predict import predict_image

app = FastAPI(title="CashVision Server - text-only response")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/recognize/", response_class=PlainTextResponse)
async def recognize_cash(image: UploadFile = File(...)):

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_ext = ".jpg"
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")
    with open(file_path, "wb") as f:
        f.write(await image.read())

    label = predict_image(file_path)

    return label

