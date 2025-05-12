import cv2
import os
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
from tqdm import tqdm
import time

TRITON_URL = "localhost:8001"
IMAGE_DIR = "images"
BATCH_SIZE = 128

MODELS = [
    {"name": "ensemble_dino_onnx", "output": "dino_embedding_vector"},
    {"name": "ensemble_resnet_embed", "output": "resnet_embedding_vector"}
]

def load_image_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def prepare_batch(image_paths):
    raw_imgs = [load_image_bytes(p) for p in image_paths]
    max_len = max(len(img) for img in raw_imgs)
    padded_imgs = [
        np.pad(np.frombuffer(img, dtype=np.uint8), (0, max_len - len(img)), 'constant')
        for img in raw_imgs
    ]
    batch_array = np.stack(padded_imgs).astype(np.uint8)  # shape = [B, N]
    return batch_array

def infer_batch(client, model_name, output_name, batch_array):
    infer_input = InferInput("raw_image", batch_array.shape, "UINT8")
    infer_input.set_data_from_numpy(batch_array)
    infer_output = InferRequestedOutput(output_name)

    results = client.infer(
        model_name=model_name,
        inputs=[infer_input],
        outputs=[infer_output]
    )
    return results.as_numpy(output_name)

def run_inference(image_dir):
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    timings = {}
    all_predictions = {}

    with grpcclient.InferenceServerClient(TRITON_URL) as client:
        for model in MODELS:
            model_name = model["name"]
            output_name = model["output"]
            predictions = []
            start_time = time.time()

            for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"🔁 {model_name}"):
                batch_paths = image_paths[i:i + BATCH_SIZE]
                batch_array = prepare_batch(batch_paths)
                embeddings = infer_batch(client, model_name, output_name, batch_array)
                for path, emb in zip(batch_paths, embeddings):
                    predictions.append((os.path.basename(path), emb))

            elapsed = time.time() - start_time
            timings[model_name] = round(elapsed, 2)
            all_predictions[model_name] = predictions

    return timings, all_predictions

# Run and compare
if __name__ == "__main__":
    timings, results = run_inference(IMAGE_DIR)

    print("\n⏱️ Inference Timings:")
    for model, t in timings.items():
        print(f"  {model}: {t}s")

    print("\n📌 First 5 results from ResNet:")
    for fname, emb in results["ensemble_resnet_embed"][:5]:
        print(f"{fname}: {emb[:5]}...")

    print("\n📌 First 5 results from DINO:")
    for fname, emb in results["ensemble_dino_onnx"][:5]:
        print(f"{fname}: {emb[:5]}...")
