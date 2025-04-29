from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from typing import List
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
import numpy as np
import time
from utils import read_and_pad_images  # still using padding for now
from dotenv import load_dotenv
import os
import json


load_dotenv()

TRITON_URL = os.getenv("TRITON_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
INPUT_NAME = os.getenv("INPUT_NAME")
OUTPUT_NAME = os.getenv("OUTPUT_NAME")
TRITON_URL_HTTP_MODEL = os.getenv("TRITON_URL_HTTP_MODEL")

app = FastAPI()
client = grpcclient.InferenceServerClient(url=TRITON_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)



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

@app.post("/infer_from_id")
async def infer_from_id(image_id: str = Form(...)):
    if not image_id:
        return JSONResponse(status_code=400, content={"error": "No image_id provided."})

    try:
        start = time.time()

        model_name = "ensemble_model_resnet"
        input_name = "image_id"
        output_name = "classification_output"

        input_tensor = InferInput(input_name, [1, 1], "BYTES")
        input_tensor.set_data_from_numpy(np.array([[image_id.encode('utf-8')]], dtype="object"))

        output_tensor = InferRequestedOutput(output_name)

        response = client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        output_data = response.as_numpy(output_name)
        top5_indices = np.argsort(output_data[0])[-5:][::-1]

        # Load class labels
        with open("labels.json", "r") as f:
            labels = json.load(f)

        # Print top-5 predictions with class names and confidences
        print("Top-5 Predicted Class Indices:", top5_indices)
        predictions = []
        for idx in top5_indices:
            class_name = labels[idx] if isinstance(labels, list) and idx < len(labels) else f"Unknown ({idx})"
            confidence = float(output_data[0][idx])
            print(f"Class: {class_name}, Confidence: {confidence:.4f}")
            predictions.append({"class_index": int(idx), "class_name": class_name, "confidence": confidence})

        end = time.time()
        latency_ms = round((end - start) * 1000, 2)

        return {
            "image_id": image_id,
            "predictions": predictions,
            "timing_ms": latency_ms
        }

    except Exception as e:
        print(f"❌ Exception in /infer_from_id: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


from pydantic import BaseModel
from typing import List

class ImageIDsRequest(BaseModel):
    image_ids: List[str]

@app.post("/infer_from_ids_dino_json")
async def infer_from_ids(request: ImageIDsRequest):
    try:
        start = time.time()

        image_id_list = request.image_ids

        model_name = "ensemble_model_dino"
        input_name = "image_id"
        output_name = "dino_embedding_vector"

        encoded_ids = np.array([[img_id.encode("utf-8")] for img_id in image_id_list], dtype=object)
        input_tensor = InferInput(input_name, [len(image_id_list), 1], "BYTES")
        input_tensor.set_data_from_numpy(encoded_ids)

        output_tensor = InferRequestedOutput(output_name)

        response = client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        output_data = response.as_numpy(output_name)
        end = time.time()
        latency_ms = round((end - start) * 1000, 2)

        return {
            "image_ids": image_id_list,
            "predictions": output_data.tolist(),
            "timing_ms": latency_ms
        }

    except Exception as e:
        print(f"❌ Exception in /infer_from_ids_dino_json: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

