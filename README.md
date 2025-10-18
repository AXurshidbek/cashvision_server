# CashVision Server

**CashVision Server** is a FastAPI-based server for **recognizing cash value from images**.  
It uses a **YOLO (Ultralytics) `.pt` model** to detect currency and returns **only a plain text response** (e.g., `"5000"`, `"10000"`). Audio output is not included; the server only returns the recognized cash value as text.

---

## Features

- YOLOv10 model for cash detection
- FastAPI REST API
- Plain text response only (no audio)
- Handles image uploads
- Swagger UI for testing
- Works locally and on a network (`0.0.0.0`)

---
## Requirements

- Python 3.10+
- pip

Required Python packages (`requirements.txt`):

fastapi
uvicorn
ultralytics
pillow
python-multipart

---

## Installation and Running

1. Clone the repository:

```bash
git clone https://github.com/Mobilening/cashvision_server.git
cd cashvision_server
Create and activate a virtual environment:

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt
Run the server locally:

bash
Copy code
python -m uvicorn main:app --reload
Run the server accessible on the network:

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
API Usage
Test via Swagger UI:
http://127.0.0.1:8000/docs

POST /recognize/
Form data: image file (image)

Response: plain text label only
Example response:
"5000"

Test with curl
curl -X POST http://127.0.0.1:8000/recognize/ -F "image=@/path/to/bill.jpg"
Notes
The model file must be located in model/cash_model.pt.

The model should be in YOLOv8 (Ultralytics) format.

Install python-multipart to enable file uploads:

pip install python-multipart
The server only returns the recognized cash value as plain text. Audio output is not included.

Mobile GitHub: https://github.com/Cbekoder/CashVisionMobile