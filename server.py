# server.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

from predict_pytorch import DogBreedPredictor
import config_pytorch as config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Predictor singleton
predictor = None
def get_predictor():
    global predictor
    if predictor is None:
        print("Loading model...")
        predictor = DogBreedPredictor()  # Tự động tải model nếu chưa có
        print("Model loaded!")
    return predictor

# ================= HEALTH CHECK =================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# ================= GET BREEDS =================
@app.route('/api/breeds', methods=['GET'])
def get_breeds():
    try:
        model = get_predictor()
        breeds = [
            {
                'index': i,
                'code': model.breed_names[i],
                'name': model.breed_names[i].split('-')[1].replace('_', ' ').title()
            }
            for i in range(len(model.breed_names))
        ]
        return jsonify({'success': True, 'total': len(breeds), 'breeds': breeds})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================= PREDICT IMAGE =================
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        top_k = int(request.form.get('top_k', 5))
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        model = get_predictor()
        result = model.predict(image, top_k=top_k)

        return jsonify({
            'success': True,
            'prediction': result['top_prediction'],
            'top_predictions': result['top_k_predictions']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================= PREDICT BASE64 =================
@app.route('/api/predict/base64', methods=['POST'])
def predict_base64():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        top_k = int(data.get('top_k', 5))
        model = get_predictor()
        result = model.predict(image, top_k=top_k)

        return jsonify({
            'success': True,
            'prediction': result['top_prediction'],
            'top_predictions': result['top_k_predictions']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================= ERROR HANDLER =================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ================= RUN =================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)