import requests
import torch
from PIL import Image
from numpy.ma.core import array
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from models.gdino.utils import DEVICE


class GDINO:
    def build_model(self, ckpt_path: str | None = None, device=DEVICE):
        model_id = "IDEA-Research/grounding-dino-base" if ckpt_path is None else ckpt_path
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        for i, prompt in enumerate(texts_prompt):
            if prompt[-1] != ".":
                texts_prompt[i] += "."
        inputs = self.processor(images=images_pil, text=texts_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in images_pil],
        )
        return results

    def predict_dino_1_6_pro(self, img, text, bbox_threshold, iou_threshold):
        headers = {
            "Content-Type": "application/json",
            "Token": ""
        }
        if headers["Token"] == "":
            raise ValueError("Please provide your token in the header.")
        resp = requests.post(
            url='https://api.deepdataspace.com/v2/task/grounding_dino/detection',
            json={"model": "GroundingDino-1.6-Pro",
                  "image": f"data:image/png;base64,{img}",
                  "prompt": {"type": "text",
                             "text": f"{text}"},
                  "targets": ["bbox"],
                  "bbox_threshold": bbox_threshold,
                  "iou_threshold": iou_threshold},

            headers=headers
        )
        json_resp = resp.json()
        task_uuid = json_resp["data"]["task_uuid"]

        while True:
            resp = requests.get(f'https://api.deepdataspace.com/v2/task_status/{task_uuid}', headers=headers)
            json_resp = resp.json()
            if json_resp["data"]["status"] not in ["waiting", "running"]:
                break
        results = json_resp['data']['result']['objects']
        predicted_results = {"scores": [], "labels": [], "boxes": []}
        for result in results:
            predicted_results["boxes"].append(result['bbox'])
            predicted_results["labels"].append(result['category'])
            predicted_results["scores"].append(result['score'])

        return [predicted_results]


if __name__ == "__main__":
    gdino = GDINO()
    gdino.build_model()
    out = gdino.predict(
        [Image.open("assets/frame_00000.jpg")],
        ["football.shoes"],
        0.3,
        0.25,
    )
    # [{'scores': tensor([...]), 'labels': [...], 'boxes': tensor([...])}]
    print(out)
