"""
Flask API Server for Dog Breed Prediction
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

from predict_pytorch import DogBreedPredictor
import config_pytorch as config

# ================== INIT ==================
app = Flask(__name__)
CORS(app)

# ================== GLOBAL MODEL ==================
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        print("Loading model (first request)...")
        predictor = DogBreedPredictor()
        print("Model loaded!")
    return predictor

# ================== ROOT (QUAN TRỌNG) ==================
@app.route('/')
def home():
    return "Dog AI API is running"

# ================== HEALTH ==================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# ================== BREEDS ==================
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

        return jsonify({
            'success': True,
            'total': len(breeds),
            'breeds': breeds
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================== PREDICT ==================
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        top_k = int(request.form.get('top_k', 5))

        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

        model = get_predictor()
        result = model.predict(image, top_k=top_k)

        return jsonify({
            'success': True,
            'prediction': result['top_prediction'],
            'top_predictions': result['top_k_predictions']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================== BASE64 ==================
@app.route('/api/predict/base64', methods=['POST'])
def predict_base64():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')

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

# ================== BATCH ==================
@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': 'No image files provided'}), 400

        image_files = request.files.getlist('images')

        if len(image_files) == 0:
            return jsonify({'success': False, 'error': 'No images provided'}), 400

        top_k = int(request.form.get('top_k', 5))

        model = get_predictor()
        results = []

        for image_file in image_files:
            try:
                image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

                result = model.predict(image, top_k=top_k)

                results.append({
                    'filename': image_file.filename,
                    'success': True,
                    'prediction': result['top_prediction'],
                    'top_predictions': result['top_k_predictions']
                })

            except Exception as e:
                results.append({
                    'filename': image_file.filename,
                    'success': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'total': len(results),
            'predictions': results
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================== ERROR ==================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ================== MAIN ==================
if __name__ == '__main__':
    print("DOG BREED API STARTING...")
    print(f"Model: {config.MODEL_ARCHITECTURE}")

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)