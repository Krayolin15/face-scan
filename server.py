from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from deepface import DeepFace

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Parse JSON request to get base64 image data
        data = request.get_json()
        img_data = data['image']

        # Remove base64 prefix (e.g., "data:image/jpeg;base64,")
        img_str = img_data.split(',')[1]

        # Decode base64 string to bytes
        img_bytes = base64.b64decode(img_str)

        # Convert bytes to numpy array for OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode numpy array into OpenCV BGR image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Analyze the image with DeepFace for emotions
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # If the result is a list (multiple faces), take the first one
        if isinstance(result, list):
            result = result[0]

        # Extract dominant emotion from the analysis result
        dominant_emotion = result['dominant_emotion']

        # Return the mood in JSON format
        return jsonify({"mood": dominant_emotion})

    except Exception as e:
        # If any error occurs, send back error message in JSON
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run Flask app in debug mode
    app.run(debug=True)
