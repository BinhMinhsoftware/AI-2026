"""
PyTorch Inference script for Dog Breed Classification
"""
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import pickle

import config_pytorch as config
from model_pytorch import DogBreedModel


class DogBreedPredictor:
    """Dog breed predictor using PyTorch"""

    def __init__(self, model_path=None, device=None):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model checkpoint
            device: torch.device (default: auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Default model path
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(config.MODEL_DIR, 'final_model.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from: {model_path}")

        # Load label mapping
        mapping_file = os.path.join(config.MODEL_DIR, 'label_mapping.pkl')
        with open(mapping_file, 'rb') as f:
            label_mapping = pickle.load(f)

        self.index_to_label = label_mapping['index_to_label']
        self.breed_names = label_mapping['breed_names']
        self.num_classes = len(self.breed_names)

        # Load model
        self.model = DogBreedModel(
            num_classes=self.num_classes,
            architecture=config.MODEL_ARCHITECTURE,
            pretrained=False
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully! (Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
        print(f"Using device: {self.device}")

        # Create transform
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])

    def preprocess_image(self, image_input):
        """
        Preprocess image for prediction

        Args:
            image_input: Can be:
                - File path (str)
                - PIL Image
                - NumPy array

        Returns:
            Preprocessed tensor
        """
        # Load image if path provided
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input

        # Apply transform
        image_tensor = self.transform(image)
        return image_tensor

    @torch.no_grad()
    def predict(self, image_input, top_k=5):
        """
        Predict dog breed from image

        Args:
            image_input: Image input
            top_k: Number of top predictions

        Returns:
            Dictionary with predictions
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_input)
        image_batch = image_tensor.unsqueeze(0).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_batch)
            probabilities = F.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        top_probs, top_indices = probabilities.topk(top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            idx = idx.item()
            breed_code = self.index_to_label[idx]
            breed_name = breed_code.split('-')[1].replace('_', ' ').title()
            confidence = prob.item()

            results.append({
                'breed': breed_name,
                'breed_code': breed_code,
                'confidence': confidence,
                'confidence_percent': confidence * 100
            })

        return {
            'top_prediction': results[0],
            'top_k_predictions': results,
            'all_probabilities': probabilities.cpu().numpy()
        }

    def predict_batch(self, image_paths, top_k=5):
        """
        Predict multiple images

        Args:
            image_paths: List of image paths
            top_k: Number of top predictions

        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path, top_k=top_k)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                print(f"Error predicting {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })

        return results

    def visualize_prediction(self, image_input, save_path=None):
        """
        Visualize prediction with image

        Args:
            image_input: Image input
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt

        # Load original image
        if isinstance(image_input, str):
            original_img = Image.open(image_input).convert('RGB')
        else:
            original_img = image_input

        # Predict
        result = self.predict(image_input, top_k=5)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Show image
        ax1.imshow(original_img)
        ax1.axis('off')
        ax1.set_title(
            f"Predicted: {result['top_prediction']['breed']}\n"
            f"Confidence: {result['top_prediction']['confidence_percent']:.2f}%",
            fontsize=14,
            weight='bold'
        )

        # Show top-5 predictions
        breeds = [p['breed'] for p in result['top_k_predictions']]
        confidences = [p['confidence_percent'] for p in result['top_k_predictions']]

        colors = ['#2E86AB' if i == 0 else '#F18F01' if i < 3 else '#C73E1D'
                  for i in range(len(breeds))]

        ax2.barh(breeds, confidences, color=colors, alpha=0.8)
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_title('Top 5 Predictions', fontsize=14, weight='bold')
        ax2.invert_yaxis()
        ax2.set_xlim(0, 100)

        # Add percentage labels
        for i, v in enumerate(confidences):
            ax2.text(v + 2, i, f'{v:.2f}%', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Example usage"""
    print("=" * 80)
    print("DOG BREED PREDICTOR (PyTorch)")
    print("=" * 80)

    # Initialize predictor
    predictor = DogBreedPredictor()

    print("\nExample Usage:")
    print("-" * 80)
    print("\n1. Predict single image:")
    print("   result = predictor.predict('path/to/dog.jpg')")
    print("   print(result['top_prediction'])")

    print("\n2. Predict with visualization:")
    print("   predictor.visualize_prediction('path/to/dog.jpg', save_path='result.png')")

    print("\n3. Batch prediction:")
    print("   images = ['img1.jpg', 'img2.jpg', 'img3.jpg']")
    print("   results = predictor.predict_batch(images)")

    print("\n" + "=" * 80)
    print("Ready to predict!")
    print("=" * 80)


if __name__ == "__main__":
    main()
