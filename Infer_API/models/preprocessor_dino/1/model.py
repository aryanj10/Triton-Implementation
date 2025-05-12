import triton_python_backend_utils as pb_utils
import numpy as np
import json
import cv2
import torch
import torchvision.transforms.v2 as transforms
import logging
import time
from concurrent.futures import ThreadPoolExecutor

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "preprocessed_image")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        # Logger
        self.logger = logging.getLogger('preprocessor_dino')
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info("✅ DINOv2 Preprocessor initialized")

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=8)  # for decoding in parallel

    def execute(self, requests):
        request = requests[0]
        responses = []

        try:
            total_start = time.time()

            input_tensor = pb_utils.get_input_tensor_by_name(request, "raw_image")
            image_byte_array = input_tensor.as_numpy()  # shape: [B, N]

            decode_start = time.time()

            # Parallel decode
            decoded_imgs = list(self.executor.map(
                self._safe_decode, [img.tobytes() for img in image_byte_array]
            ))
            decode_end = time.time()

            transform_start = time.time()
            preprocessed_batch = []
            for img in decoded_imgs:
                preprocessed_batch.append(self.transform(img.to(self.device)))
            transform_end = time.time()

            batch_tensor = torch.stack(preprocessed_batch)
            output_tensor = pb_utils.Tensor("preprocessed_image", batch_tensor.cpu().numpy())
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

            self.logger.info(f"✅ Batch size: {len(preprocessed_batch)} | Total time: {round(time.time() - total_start, 4)}s")
            self.logger.info(f"⏱ Avg decode: {(decode_end - decode_start)/len(preprocessed_batch):.4f}s | transform: {(transform_end - transform_start)/len(preprocessed_batch):.4f}s")

        except Exception as e:
            self.logger.error(f"❌ Fatal error in execute: {e}")
            dummy_tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)
            output_tensor = pb_utils.Tensor("preprocessed_image", dummy_tensor)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def _safe_decode(self, img_bytes):
        try:
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV failed to decode image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            self.logger.error(f"❌ Decode error: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32)

    def finalize(self):
        self.logger.info("Preprocessor unloaded")
        self.executor.shutdown()
        self.logger.info("✅ DINOv2 Preprocessor finalized")
