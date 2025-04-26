from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import httpx

from typing import List
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
import numpy as np
import time
from utils import read_and_pad_images  # still using padding for now
from dotenv import load_dotenv
import os

load_dotenv()

TRITON_URL = os.getenv("TRITON_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
INPUT_NAME = os.getenv("INPUT_NAME")
OUTPUT_NAME = os.getenv("OUTPUT_NAME")
TRITON_URL_HTTP_MODEL = os.getenv("TRITON_URL_HTTP_MODEL")

app = FastAPI()
client = grpcclient.InferenceServerClient(url=TRITON_URL)




@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded."})
    
    try:
        # Start timing
        start = time.time()

        # Read and preprocess image bytes into numpy padded array
        batch_array = read_and_pad_images(files)

        # Set Triton input
        input_tensor = InferInput(INPUT_NAME, batch_array.shape, "UINT8")
        input_tensor.set_data_from_numpy(batch_array)

        # Set requested output
        output_tensor = InferRequestedOutput(OUTPUT_NAME)

        # Run inference
        response = client.infer(
            model_name=MODEL_NAME,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        # Extract predictions
        output_data = response.as_numpy(OUTPUT_NAME)
        preds = np.argmax(output_data, axis=1)

        # End timing
        end = time.time()
        total_sec = round(end - start, 4)
        avg_latency = round((total_sec / len(files)) * 1000, 2)

        results = [{"filename": file.filename, "predicted_cluster": int(pred)} for file, pred in zip(files, preds)]

        return {
            "results": results,
            "timing": {
                "total_images": len(files),
                "total_sec": total_sec,
                "average_latency_per_image_ms": avg_latency
            }
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
