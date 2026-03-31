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
CORS(app)  # Enable CORS for all routes

# Initialize predictor (load model once at startup)
print("Initializing Dog Breed Predictor...")
predictor = DogBreedPredictor()
print("Predictor ready!")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'num_classes': len(predictor.breed_names)
    })


@app.route('/api/breeds', methods=['GET'])
def get_breeds():
    """Get list of all supported dog breeds"""
    breeds = [
        {
            'index': i,
            'code': predictor.breed_names[i],
            'name': predictor.breed_names[i].split('-')[1].replace('_', ' ').title()
        }
        for i in range(len(predictor.breed_names))
    ]
    return jsonify({
        'total': len(breeds),
        'breeds': breeds
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict dog breed from uploaded image

    Request:
        - Content-Type: multipart/form-data
        - Field: image (file)
        - Optional: top_k (int, default=5)

    Response:
        {
            "success": true,
            "prediction": {
                "breed": "Golden Retriever",
                "confidence": 0.95,
                "confidence_percent": 95.0
            },
            "top_predictions": [
                {"breed": "...", "confidence": 0.95},
                ...
            ]
        }
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        image_file = request.files['image']

        # Check if file is empty
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400

        # Get top_k parameter
        top_k = int(request.form.get('top_k', 5))

        # Read image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Predict
        result = predictor.predict(image, top_k=top_k)

        # Format response
        response = {
            'success': True,
            'prediction': result['top_prediction'],
            'top_predictions': result['top_k_predictions']
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict/base64', methods=['POST'])
def predict_base64():
    """
    Predict dog breed from base64 encoded image

    Request:
        {
            "image": "data:image/jpeg;base64,...",
            "top_k": 5
        }

    Response:
        Same as /api/predict
    """
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400

        # Parse base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get top_k parameter
        top_k = int(data.get('top_k', 5))

        # Predict
        result = predictor.predict(image, top_k=top_k)

        # Format response
        response = {
            'success': True,
            'prediction': result['top_prediction'],
            'top_predictions': result['top_k_predictions']
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple images at once

    Request:
        - Content-Type: multipart/form-data
        - Fields: images[] (multiple files)
        - Optional: top_k (int, default=5)

    Response:
        {
            "success": true,
            "predictions": [
                {
                    "filename": "dog1.jpg",
                    "prediction": {...},
                    "top_predictions": [...]
                },
                ...
            ]
        }
    """
    try:
        # Check if images are in request
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image files provided'
            }), 400

        image_files = request.files.getlist('images')

        if len(image_files) == 0:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400

        # Get top_k parameter
        top_k = int(request.form.get('top_k', 5))

        # Process each image
        results = []
        for image_file in image_files:
            try:
                # Read image
                image_bytes = image_file.read()
                image = Image.open(io.BytesIO(image_bytes))

                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Predict
                result = predictor.predict(image, top_k=top_k)

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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("=" * 80)
    print("DOG BREED PREDICTION API SERVER")
    print("=" * 80)
    print(f"Model: {config.MODEL_ARCHITECTURE}")
    print(f"Number of breeds: {len(predictor.breed_names)}")
    print("\nAvailable endpoints:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/breeds          - List all breeds")
    print("  POST /api/predict         - Predict single image (multipart)")
    print("  POST /api/predict/base64  - Predict single image (base64)")
    print("  POST /api/predict/batch   - Predict multiple images")
    print("\nStarting server on http://localhost:5000")
    print("=" * 80)

    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)