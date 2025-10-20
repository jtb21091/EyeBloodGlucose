from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib, numpy as np, cv2
from PIL import Image
import io

app = FastAPI()

# Allow local access + later Shopify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_index():
    return FileResponse("index.html")  # <--- serve this file directly

# ---- prediction endpoint ----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    feat = arr.flatten().reshape(1, -1)

    model = joblib.load("best_model.pkl")
    pred = float(model.predict(feat)[0])
    return {"estimate_mg_dl": pred}
