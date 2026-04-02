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

        # ================= DEVICE =================
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # ================= MODEL PATH =================
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')

        # ❌ KHÔNG DOWNLOAD NỮA
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please upload best_model.pth to models_pytorch/"
            )

        print(f"Loading model from: {model_path}")

        # ================= LOAD LABEL MAPPING =================
        mapping_file = os.path.join(config.MODEL_DIR, 'label_mapping.pkl')

        if not os.path.exists(mapping_file):
            raise FileNotFoundError(
                f"Label mapping not found at {mapping_file}"
            )

        with open(mapping_file, 'rb') as f:
            label_mapping = pickle.load(f)

        self.index_to_label = label_mapping['index_to_label']
        self.breed_names = label_mapping['breed_names']
        self.num_classes = len(self.breed_names)

        print(f"Loaded {self.num_classes} classes")

        # ================= LOAD MODEL =================
        self.model = DogBreedModel(
            num_classes=self.num_classes,
            architecture=config.MODEL_ARCHITECTURE,
            pretrained=False
        )

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully! (Val Acc: {checkpoint.get('val_acc', 'N/A')})")

        # ================= TRANSFORM =================
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])

    # ================= PREPROCESS =================
    def preprocess_image(self, image_input):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = Image.open(image_input).convert('RGB')

        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)

        else:
            image = image_input

        return self.transform(image)

    # ================= PREDICT =================
    @torch.no_grad()
    def predict(self, image_input, top_k=5):

        image_tensor = self.preprocess_image(image_input)
        image_batch = image_tensor.unsqueeze(0).to(self.device)

        outputs = self.model(image_batch)
        probabilities = F.softmax(outputs, dim=1)[0]

        top_probs, top_indices = probabilities.topk(top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            idx = idx.item()

            breed_code = self.index_to_label[idx]
            breed_name = breed_code.split('-')[1].replace('_', ' ').title()

            results.append({
                'breed': breed_name,
                'breed_code': breed_code,
                'confidence': prob.item(),
                'confidence_percent': prob.item() * 100
            })

        return {
            'top_prediction': results[0],
            'top_k_predictions': results,
            'all_probabilities': probabilities.cpu().numpy()
        }

    # ================= BATCH =================
    def predict_batch(self, image_paths, top_k=5):
        results = []

        for img_path in image_paths:
            try:
                result = self.predict(img_path, top_k=top_k)
                result['image_path'] = img_path
                results.append(result)

            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })

        return results

    # ================= VISUALIZE =================
    def visualize_prediction(self, image_input, save_path=None):
        import matplotlib.pyplot as plt

        if isinstance(image_input, str):
            original_img = Image.open(image_input).convert('RGB')
        else:
            original_img = image_input

        result = self.predict(image_input, top_k=5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.imshow(original_img)
        ax1.axis('off')
        ax1.set_title(
            f"{result['top_prediction']['breed']} "
            f"({result['top_prediction']['confidence_percent']:.2f}%)"
        )

        breeds = [p['breed'] for p in result['top_k_predictions']]
        confidences = [p['confidence_percent'] for p in result['top_k_predictions']]

        ax2.barh(breeds, confidences)
        ax2.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()


# ================= MAIN =================
def main():
    predictor = DogBreedPredictor()
    print("Ready!")


if __name__ == "__main__":
    main()