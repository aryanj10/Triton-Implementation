"""import os
import time
import numpy as np
import cv2
from tqdm import tqdm
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput

# === Config ===
TRITON_URL = "localhost:8001"
MODEL_NAME = "ensemble_model"  # Ensemble: preprocessor + classifier
IMAGE_DIR = "/home/aj3246/material/material_data/1"
BATCH_SIZE = 256
INPUT_NAME = "RAW_IMAGE"
OUTPUT_NAME = "output__0"
IMAGE_SIZE = (224, 224)

# === Load and preprocess all images ===
image_paths = [
    os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"🔍 Found {len(image_paths)} images. Preprocessing...")

# Load and resize all images
all_images = []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_images.append(img)

all_images = np.array(all_images, dtype=np.uint8)  # shape = [N, 224, 224, 3]

# === Setup Triton client ===
client = grpcclient.InferenceServerClient(url=TRITON_URL)

# === Inference in batches ===
total_time = 0
start = time.time()

all_predictions = []

for i in tqdm(range(0, len(all_images), BATCH_SIZE), desc="🔁 Sending batches"):
    batch = all_images[i:i+BATCH_SIZE]  # shape: [B, 224, 224, 3]

    input_tensor = InferInput(INPUT_NAME, batch.shape, "UINT8")
    input_tensor.set_data_from_numpy(batch)

    output_tensor = InferRequestedOutput(OUTPUT_NAME)

    t0 = time.time()
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    t1 = time.time()
    total_time += (t1 - t0)

    # === Extract predictions for this batch
    output_data = response.as_numpy(OUTPUT_NAME)
    predicted_classes = np.argmax(output_data, axis=1)

    # Store image name and predicted class
    for j, pred in enumerate(predicted_classes):
        image_index = i + j
        image_path = image_paths[image_index]
        all_predictions.append((os.path.basename(image_path), int(pred)))

# === Report
end = time.time()
n_images = len(all_images)

print("\n✅ Benchmark Complete!")
print(f"🖼️ Total images processed: {n_images}")
print(f"⏱️ Total time (inference only): {total_time:.4f} sec")
print(f"📊 Average latency per image: {total_time / n_images * 1000:.2f} ms")
print(f"⚡ Total end-to-end time: {end - start:.4f} sec")

print(all_predictions)
print(f"🔢 Total predictions collected: {len(all_predictions)}")
"""


import os
import time
import numpy as np
from tqdm import tqdm
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput

# === Config ===
TRITON_URL = "localhost:8001"
MODEL_NAME = "ensemble_model"  # Ensemble: preprocessor + classifier
IMAGE_DIR = "/home/aj3246/material/material_data/1"
BATCH_SIZE = 1000
INPUT_NAME = "RAW_IMAGE"
OUTPUT_NAME = "output__0"

# === Find all images ===
image_paths = [
    os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"🔍 Found {len(image_paths)} images. Loading raw bytes...")

# === Load images as raw bytes ===
def load_image(img_path: str):
    """Loads an image as an array of bytes."""
    return np.fromfile(img_path, dtype="uint8")

# === Setup Triton client ===
client = grpcclient.InferenceServerClient(
    url=TRITON_URL,
    verbose=False,
    ssl=False,
)


# === Inference one by one ===
total_time = 0
start = time.time()
all_predictions = []


all_predictions = []
total_time = 0
start = time.time()

for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="🔁 Sending batches"):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    
    # Load and pad raw image bytes
    raw_imgs = [load_image(p) for p in batch_paths]
    max_len = max(img.shape[0] for img in raw_imgs)
    padded_imgs = [np.pad(img, (0, max_len - len(img)), 'constant') for img in raw_imgs]
    batch_array = np.stack(padded_imgs).astype(np.uint8)  # shape = [B, N]
    
    # Set input
    input_tensor = InferInput(INPUT_NAME, batch_array.shape, "UINT8")
    input_tensor.set_data_from_numpy(batch_array)
    output_tensor = InferRequestedOutput(OUTPUT_NAME)

    # Inference
    t0 = time.time()
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    t1 = time.time()
    total_time += (t1 - t0)

    # Parse output
    output_data = response.as_numpy(OUTPUT_NAME)
    preds = np.argmax(output_data, axis=1)
    
    for path, pred in zip(batch_paths, preds):
        all_predictions.append((os.path.basename(path), int(pred)))



# === Report
end = time.time()
n_images = len(image_paths)

print("\n✅ Benchmark Complete!")
print(f"🖼️ Total images processed: {n_images}")
print(f"⏱️ Total time (inference only): {total_time:.4f} sec")
print(f"📊 Average latency per image: {total_time / n_images * 1000:.2f} ms")
print(f"⚡ Total end-to-end time: {end - start:.4f} sec")
print(f"🔢 Total predictions collected: {len(all_predictions)}")

# === Print sample predictions ===
print("\n📊 Sample predictions (first 10):")
for i, (img_name, pred) in enumerate(all_predictions[:10]):
    print(f"{img_name}: Class {pred}")