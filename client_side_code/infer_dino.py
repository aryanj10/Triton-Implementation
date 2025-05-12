import cv2
import os
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
from tqdm import tqdm

TRITON_URL = "localhost:8001"
MODEL_NAME = "ensemble_dino_onnx"
INPUT_NAME = "raw_image"
LENGTH_NAME = "image_length"
OUTPUT_NAME = "dino_embedding_vector"
BATCH_SIZE = 128
IMAGE_DIR = "images"

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


def infer_batch(client, batch_array):
    infer_input = InferInput(INPUT_NAME, batch_array.shape, "UINT8")
    infer_input.set_data_from_numpy(batch_array)

    infer_output = InferRequestedOutput(OUTPUT_NAME)

    results = client.infer(
        model_name=MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output]
    )

    return results.as_numpy(OUTPUT_NAME)


def run_inference(image_dir):
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    predictions = []
    with grpcclient.InferenceServerClient(TRITON_URL) as client:
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="🔁 Sending Batches"):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            batch_array = prepare_batch(batch_paths)
            embeddings = infer_batch(client, batch_array)


            for path, emb in zip(batch_paths, embeddings):
                predictions.append((os.path.basename(path), emb))
    return predictions

# Usage
if __name__ == "__main__":
    results = run_inference(IMAGE_DIR)
    for fname, emb in results[:5]:
        print(f"{fname}: {emb[:5]}...")  # Print first 5 values of embedding
