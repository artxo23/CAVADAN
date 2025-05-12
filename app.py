from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from gtts import gTTS
from PIL import Image
import torch
import os
import zipfile

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
try:
    BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
    BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
    FLAN_TOKENIZER = T5Tokenizer.from_pretrained("google/flan-t5-large", legacy=True)
    FLAN_MODEL = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(DEVICE)
except Exception as e:
    print(f"Error loading models: {e}")
    BLIP_PROCESSOR, BLIP_MODEL, FLAN_TOKENIZER, FLAN_MODEL = None, None, None, None


@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    if not BLIP_PROCESSOR or not BLIP_MODEL or not FLAN_TOKENIZER or not FLAN_MODEL:
        return jsonify({"error": "Models could not be loaded. Please check the server logs."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    prompt = request.form.get('prompt', '').strip()

    try:
        # Save the original image
        image = Image.open(image_file).convert("RGB")
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg')
        image.save(original_image_path)
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    try:
        # Generate BLIP caption
        inputs = BLIP_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
        blip_output = BLIP_MODEL.generate(**inputs)
        blip_caption = BLIP_PROCESSOR.tokenizer.decode(blip_output[0], skip_special_tokens=True)

        # Refine caption with FLAN if a prompt is provided
        final_caption = blip_caption
        if prompt:
            flan_input = f"{prompt} {blip_caption}"
            flan_inputs = FLAN_TOKENIZER.encode(flan_input, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
            flan_output = FLAN_MODEL.generate(flan_inputs, max_length=128, num_beams=5, early_stopping=True)
            final_caption = FLAN_TOKENIZER.decode(flan_output[0], skip_special_tokens=True)

        # Generate audio file for the caption
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp3')
        tts = gTTS(final_caption)
        tts.save(audio_path)

        # Prepare outputs as a ZIP file
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(original_image_path, 'original_image.jpg')
            zipf.writestr('caption.txt', final_caption)
            zipf.writestr('user_input.txt', f"Uploaded Image: original_image.jpg\nPrompt: {prompt}")
            zipf.write(audio_path, 'output.mp3')

        return jsonify({
            "caption": final_caption,
            "image_url": f"/uploads/original_image.jpg",
            "audio_url": f"/uploads/output.mp3",
            "zip_url": f"/uploads/output.zip"
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or default to 5000
    app.run(host="0.0.0.0", port=port)