"""import streamlit as st
import numpy as np
from PIL import Image
import tritonclient.http as httpclient
import torchvision.transforms as transforms

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8000")

st.title("🧠 Cluster Classifier via Triton")
st.caption("Upload up to 100 images and get their predicted cluster numbers.")

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 100000:
        st.warning("⚠️ Please upload up to 100 images only.")
    else:
        images = []
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            images.append(image)

        # Preprocess and create batch
        input_batch = np.stack([transform(img).numpy() for img in images])  # shape: [N, 3, 224, 224]

        # Triton input
        infer_input = httpclient.InferInput("input__0", input_batch.shape, "FP32")
        infer_input.set_data_from_numpy(input_batch)

        infer_output = httpclient.InferRequestedOutput("output__0")

        # Run inference
        response = client.infer("classifier", inputs=[infer_input], outputs=[infer_output])
        predictions = response.as_numpy("output__0")
        predicted_classes = np.argmax(predictions, axis=1)

        # Show results
        st.success("✅ Inference complete!")
        for img, cluster_id in zip(images, predicted_classes):
            st.image(img, caption=f"Predicted Cluster: {cluster_id}", use_column_width=True)
"""



import os
import time
import numpy as np
import cv2
from tqdm import tqdm
import tritonclient.http as httpclient
from tritonclient.http import InferInput, InferRequestedOutput

# === Config ===
TRITON_URL = "localhost:8000"  # HTTP port
MODEL_NAME = "ensemble_model"
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

all_images = []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_images.append(img)

all_images = np.array(all_images, dtype=np.uint8)  # shape: [N, 224, 224, 3]

# === Setup HTTP client ===
client = httpclient.InferenceServerClient(url=TRITON_URL)

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

    output_data = response.as_numpy(OUTPUT_NAME)
    predicted_classes = np.argmax(output_data, axis=1)

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

print(f"🔢 Total predictions collected: {len(all_predictions)}")
