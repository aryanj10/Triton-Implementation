import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# URLs
ID_LIST_URL = "http://m3kube.urcf.drexel.edu:8079/api/get-all-images"
BASE_URL = "http://m3kube.urcf.drexel.edu:8079"
SAVE_DIR = "images"
MAX_WORKERS = 16

os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: Fetch list of image metadata
try:
    response = requests.get(ID_LIST_URL)
    response.raise_for_status()
    image_entries = response.json()  # list of dicts
except Exception as e:
    print(f"❌ Failed to fetch image list: {e}")
    exit(1)

print(f"📦 Found {len(image_entries)} images to download.")

# Step 2: Download function
def download_image(entry):
    try:
        image_id = entry["image_id"]
        filename = entry["filename"]
        img_url = f"{BASE_URL}/api/get-image/{image_id}"  # ✅ corrected

        response = requests.get(img_url, timeout=5)
        response.raise_for_status()

        save_path = os.path.join(SAVE_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(response.content)

        return True
    except Exception as e:
        return (entry["filename"], str(e))


# Step 3: Parallel download
failed = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_image, entry): entry for entry in image_entries}
    for future in tqdm(as_completed(futures), total=len(futures), desc="📥 Downloading images"):
        result = future.result()
        if result is not True:
            failed.append(result)

# Step 4: Report
if failed:
    print(f"\n⚠️ Failed to download {len(failed)} images:")
    for entry, error in failed[:10]:
        print(f" - {entry['filename']}: {error}")
    with open("failed_downloads.log", "w") as f:
        for entry, error in failed:
            f.write(f"{entry['image_id']},{entry['filename']},{error}\n")
else:
    print("\n✅ All images downloaded successfully!")
