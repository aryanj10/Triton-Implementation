import requests
import json
import time
from tqdm import tqdm

# 1. Get all image metadata
print("📡 Fetching image metadata...")
url_all_images = "http://m3kube.urcf.drexel.edu:8079/api/get-all-images"
response = requests.get(url_all_images)
response.raise_for_status()
images_data = response.json()

# 2. Extract image_ids
print(f"🧹 Extracting image IDs from {len(images_data)} entries...")
image_ids = [entry["image_id"] for entry in images_data]
image_ids = image_ids[:1000]  # Limit to 1000 for testing
print(f"🧹 Extracted {len(image_ids)} image IDs (Limiting for testing)")
# 3. Setup inference
headers = {"Content-Type": "application/json"}
inference_url = "http://localhost:8081/infer_from_ids_dino_json"
batch_size = 128
all_results = {}

# 4. Send requests in batches
print(f"🚀 Running inference in batches of {batch_size}...")
start_time = time.time()

for i in tqdm(range(0, len(image_ids), batch_size), desc="Inferencing"):
    batch = image_ids[i:i+batch_size]
    payload = {"image_ids": batch}
    resp = requests.post(inference_url, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    batch_predictions = resp.json()["predictions"]
    all_results.update(batch_predictions)

end_time = time.time()
elapsed = round(end_time - start_time, 2)

# 5. Print output
print(f"\n✅ Inference completed in {elapsed} seconds")
#print(json.dumps(all_results, indent=2))
