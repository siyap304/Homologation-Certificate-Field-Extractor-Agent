from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os
from tenacity import retry, wait_exponential, stop_after_attempt

class OCR_JSON:

    def __init__(self):
        load_dotenv()
        
        self.endpoint = os.getenv("AZURE_AI_VISION_ENDPOINT")
        self.key = os.getenv("AZURE_AI_VISION_KEY")
        
        self.client = ImageAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
    def analyze_with_retry(self, client, image):
        return client.analyze(
            image_data=image,
            visual_features=[VisualFeatures.READ]
        )

    def process_images(self, image_paths):
        final_result = {"readResult": {"blocks": []}}

        for i, image_path in enumerate(image_paths):
            with open(image_path, "rb") as image_stream:
                image = image_stream.read()

            try:
                result = self.analyze_with_retry(self.client, image)
            except Exception as e:
                print(f"Failed to analyze image {image_path}: {e}")

            ocr_json = result.as_dict()

            per_image_block = {
            "page": i+1,
            "lines": []
            }

            for block in ocr_json.get("readResult", {}).get("blocks", []):
                per_image_block["lines"].extend(block.get("lines", []))
                
            final_result["readResult"]["blocks"].append(per_image_block)

        return final_result
