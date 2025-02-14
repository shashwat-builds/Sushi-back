from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import sys
import json
from PIL import Image

def main(image_path, question):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

    # Open the image using PIL
    image = Image.open(image_path)

    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    print(json.dumps({"answer": answer}))

if __name__ == "__main__":
    image_path = sys.argv[1]
    question = sys.argv[2]
    main(image_path, question)