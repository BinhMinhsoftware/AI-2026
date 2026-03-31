"""
Flask API Server for Dog Breed Prediction
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64

from predict_pytorch import DogBreedPredictor
import config_pytorch as config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ================== FIX QUAN TRỌNG ==================
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        print("Loading model...")
        predictor = DogBreedPredictor()
        print("Model loaded!")
    return predictor

# 👉 Nếu chạy LOCAL thì load sẵn
if os.environ.get("RENDER") is None:
    print("Initializing Dog Breed Predictor (LOCAL)...")
    predictor = DogBreedPredictor()
    print("Predictor ready!")
# ===================================================


@app.route('/api/health', methods=['GET'])
def health_check():
    model = get_predictor()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model.model is not None,
        'num_classes': len(model.breed_names)
    })


@app.route('/api/breeds', methods=['GET'])
def get_breeds():
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
        'total': len(breeds),
        'breeds': breeds
    })


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


@app.route('/api/predict/base64', methods=['POST'])
def predict_base64():
    try:
        data = request.get_json()

        if 'image' not in data:
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


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': 'No image files provided'}), 400

        image_files = request.files.getlist('images')

        if len(image_files) == 0:
            return jsonify({'success': False, 'error': 'No images provided'}), 400

        top_k = int(request.form.get('top_k', 5))

        results = []
        model = get_predictor()

        for image_file in image_files:
            try:
                image_bytes = image_file.read()
                image = Image.open(io.BytesIO(image_bytes))

                if image.mode != 'RGB':
                    image = image.convert('RGB')

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


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("DOG BREED PREDICTION API SERVER")
    print("=" * 80)
    print(f"Model: {config.MODEL_ARCHITECTURE}")

    model = get_predictor()
    print(f"Number of breeds: {len(model.breed_names)}")

    print("\nStarting server...")

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)