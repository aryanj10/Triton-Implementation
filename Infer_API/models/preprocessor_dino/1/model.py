import triton_python_backend_utils as pb_utils
import numpy as np
import json
import cv2
import torch
import torchvision.transforms.v2 as transforms
import requests
import logging
from concurrent.futures import ThreadPoolExecutor

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "preprocessed_image")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        # Logger
        self.logger = logging.getLogger('preprocessor_dino')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.info("✅ DINOv2 Preprocessor initialized")

        # Image source API
        self.api_url_base = "http://m3kube.urcf.drexel.edu:8079/api/get-image/"

        # DINOv2-specific preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    def _safe_fetch_and_preprocess(self, image_id_str):
        try:
            img_bytes = self._fetch_image_from_api(image_id_str)
            img_tensor = self._decode_opencv(img_bytes)
            img_tensor = img_tensor.to(self.device)
            return self.transform(img_tensor)
        
        except Exception as e:
            self.logger.error(f"Error for {image_id_str}: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32)

    def execute(self, requests):
        request = requests[0]
        responses = []

        try:
            input_id_tensor = pb_utils.get_input_tensor_by_name(request, "image_id")
            image_id_batch = input_id_tensor.as_numpy().squeeze(axis=1)

            # Decode bytes
            image_ids = [id_bytes.decode("utf-8") for id_bytes in image_id_batch]

            # Fetch & preprocess in parallel
            with ThreadPoolExecutor(max_workers=16) as executor:
                preprocessed_batch = list(executor.map(self._safe_fetch_and_preprocess, image_ids))

            batch_tensor = torch.stack(preprocessed_batch)
            output_tensor = pb_utils.Tensor("preprocessed_image", batch_tensor.cpu().numpy())
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        except Exception as e:
            self.logger.error(f"❌ Fatal request error: {e}")
            dummy_tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)
            output_tensor = pb_utils.Tensor("preprocessed_image", dummy_tensor)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses


    def _fetch_image_from_api(self, image_id):
        url = self.api_url_base + image_id
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.content

    def _decode_opencv(self, img_bytes):
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def finalize(self):
        self.logger.info("Preprocessor unloaded")
