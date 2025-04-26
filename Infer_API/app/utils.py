import numpy as np
from typing import List
from fastapi import UploadFile

def read_and_pad_images(files: List[UploadFile]):
    """Read images as raw bytes, pad them, and return a batch array."""
    raw_imgs = [np.frombuffer(f.file.read(), dtype=np.uint8) for f in files]
    max_len = max(len(img) for img in raw_imgs)
    padded = [np.pad(img, (0, max_len - len(img)), mode='constant') for img in raw_imgs]
    batch_array = np.stack(padded).astype(np.uint8)
    return batch_array
