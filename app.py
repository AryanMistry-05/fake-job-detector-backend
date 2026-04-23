import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import os
import cv2
import pytesseract
from PIL import Image

app = Flask(__name__)

total_jobs = 0
fake_jobs = 0
real_jobs = 0
CORS(app)

pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract\tesseract.exe"

model = pickle.load(open("job_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

def detect_logo(uploaded_image):

    logos_folder = "../images"

    img = cv2.imread(uploaded_image,0)

    for logo_file in os.listdir(logos_folder):

        logo_path = os.path.join(logos_folder,logo_file)

        template = cv2.imread(logo_path,0)

        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        threshold = 0.7

        if (result >= threshold).any():
            return True

    return False
    
def check_link(url):

    url = url.lower()

    suspicious = ["bit.ly","tinyurl","whatsapp","telegram"]

    for word in suspicious:
        if word in url:
            return "⚠ Suspicious Shortened Link"

    if ".gov.in" in url:
        return "✔ Official Government Domain"

    if ".com" in url or ".net" in url:
        return "⚠ Non-government domain – verify carefully"

    return "Unknown Source"

def check_email(text):
    # Try to find a full email first
    emails = re.findall(r'\S+@\S+', text)
    
    if emails:
        for email in emails:
            if ".gov.in" in email or ".nic.in" in email:
                return "✔ Official Government Email"
            else:
                return "⚠ Suspicious Email Domain"
    
    # NEW: If no @ found, check if they just pasted a domain
    if ".gov.in" in text or ".nic.in" in text:
        return "✔ Official Government Domain detected"
        
    return "No email found"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # your model prediction
        prediction = model.predict([text])[0]
        prob = model.predict_proba([text])[0][1]

        return jsonify({
            "fake_probability": int(prob * 100),
            "trust_score": int((1 - prob) * 100),
            "trust_message": "Analysis complete",
            "email_check": "Looks Safe",
            "reasons": ["Basic analysis working"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/scanposter", methods=["POST"])
def scanposter():

    try:
        file = request.files.get("image")

        if file is None:
            return jsonify({"error": "No image uploaded"})

        filepath = "temp.png"
        file.save(filepath)

        img = Image.open(filepath)

        text = pytesseract.image_to_string(img)
        logo_found = detect_logo(filepath)

        return jsonify({
            "extracted_text": text,
            "govt_logo_detected": logo_found
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })
    
@app.route('/scanlink', methods=['POST'])
def scanlink():
    try:
        data = request.get_json()
        url = data.get("url", "").strip().lower()

        if not url:
            return jsonify({"result": "Please enter a link"})

        # List of suspicious keywords/shorteners
        suspicious_patterns = ["bit.ly", "tinyurl", "t.me", "wa.me", "whatsapp"]

        # 1. Check for suspicious patterns first (High Priority)
        if any(pattern in url for pattern in suspicious_patterns):
            return jsonify({"result": "⚠ Suspicious Shortened or Chat Link"})

        # 2. Check for official gov domains
        if ".gov.in" in url or ".nic.in" in url:
            return jsonify({"result": "✅ Government Website (Highly Trusted)"})

        # 3. Check protocol safety
        if url.startswith("http://") and not url.startswith("https://"):
            return jsonify({"result": "⚠ Suspicious (No HTTPS - Insecure)"})

        # 4. Check for generic domains
        if ".com" in url or ".net" in url or ".org" in url:
            return jsonify({"result": "⚠ Private Domain - Verify company authenticity"})

        return jsonify({"result": "Likely Safe Link"})

    except Exception as e:
        return jsonify({"result": "Error scanning link", "error": str(e)})

@app.route("/stats")
def stats():

    return jsonify({
        "total": total_jobs,
        "fake": fake_jobs,
        "real": real_jobs
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
