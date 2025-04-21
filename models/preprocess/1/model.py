import triton_python_backend_utils as pb_utils
import numpy as np
import json
import cv2
import torch
import torchvision.transforms.v2 as transforms
from concurrent.futures import ThreadPoolExecutor

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "PREPROCESSED_IMAGE"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )

        # Preprocessing pipeline with GPU support
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=8)

    def execute(self, requests):
        responses = []

        for request in requests:
            raw_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
            raw_bytes = raw_tensor.as_numpy()

            if len(raw_bytes.shape) == 1:  # single image
                imgs = [self._decode_opencv(raw_bytes)]
            else:
                imgs = list(self.executor.map(self._decode_opencv, raw_bytes))

            tensor_batch = torch.stack(imgs).to(self.device)
            processed_batch = self.transform(tensor_batch)  # GPU-accelerated
            processed_batch = processed_batch.cpu().numpy().astype(self.output_dtype)

            output_tensor = pb_utils.Tensor("PREPROCESSED_IMAGE", processed_batch)
            responses.append(pb_utils.InferenceResponse([output_tensor]))

        return responses

    def _decode_opencv(self, img_bytes):
        try:
            img_np = np.frombuffer(img_bytes.tobytes(), dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            print(f"❌ Error decoding image: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32)

    def finalize(self):
        print("Preprocessing model unloaded")
        self.executor.shutdown()
