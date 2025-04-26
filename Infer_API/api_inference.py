import requests
import os
from math import ceil

API_URL = "https://cdd7-144-118-78-26.ngrok-free.app/predict"
IMG_FOLDER = "material/material_data/1"
BATCH_SIZE = 1000

# Gather all image paths
all_images = [
    os.path.join(IMG_FOLDER, fname)
    for fname in os.listdir(IMG_FOLDER)
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
]

total_images = len(all_images)
print(f"📦 Found {total_images} images. Sending in batches of {BATCH_SIZE}...")

# Split and send in chunks
for i in range(0, total_images, BATCH_SIZE):
    batch_paths = all_images[i:i + BATCH_SIZE]
    files = [("files", (os.path.basename(path), open(path, "rb"), "image/jpeg")) for path in batch_paths]

    print(f"🔁 Sending batch {i // BATCH_SIZE + 1} with {len(files)} images...")
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        predictions = response.json()["results"]
        for pred in predictions:
            print(f"{pred['filename']} → Cluster {pred['predicted_cluster']}")

        timings= response.json()["timing"]
        print(f"⏱️ Timing: {timings['total_images']} images processed in {timings['total_sec']} seconds.")
        print(f"Average latency per image: {timings['average_latency_per_image_ms']} ms")
        print("✅ Batch processed successfully!")
    else:
        print("❌ Error:", response.status_code, response.text)