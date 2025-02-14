from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

@app.route("/vqa", methods=["POST"])
def vqa():
    if "image" not in request.files or "question" not in request.form:
        return jsonify({"error": "Missing image or question"}), 400

    # Load image
    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read()))

    # Get question
    question = request.form["question"]

    # Process image and question
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
