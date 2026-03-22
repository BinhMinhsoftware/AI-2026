"""
PyTorch Data loading and preprocessing utilities
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

import config_pytorch as config


class DogBreedDataset(Dataset):
    """PyTorch Dataset for Dog Breed Classification"""

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of labels (class indices)
            transform: torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


class DogBreedDataLoader:
    """Data loader manager for Stanford Dogs Dataset"""

    def __init__(self):
        self.dataset_dir = config.DATASET_DIR
        self.img_height = config.IMG_HEIGHT
        self.img_width = config.IMG_WIDTH
        self.batch_size = config.BATCH_SIZE

        # Get breed names
        self.breed_names = sorted([
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
        ])
        self.num_classes = len(self.breed_names)

        # Create label mapping
        self.label_to_index = {label: idx for idx, label in enumerate(self.breed_names)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        print(f"Found {self.num_classes} dog breeds")

    def load_image_paths(self, use_cache=True):
        """Load all image paths and labels"""
        cache_file = os.path.join(config.BASE_DIR, 'image_paths_cache.pkl')

        # Try to load from cache
        if use_cache and os.path.exists(cache_file):
            print("Loading image paths from cache...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return data['image_paths'], data['labels']

        print("Scanning dataset for images...")
        image_paths = []
        labels = []

        for breed_folder in tqdm(self.breed_names, desc="Loading breeds"):
            breed_path = os.path.join(self.dataset_dir, breed_folder)
            label_index = self.label_to_index[breed_folder]

            # Get all images in breed folder
            image_files = [f for f in os.listdir(breed_path) if f.endswith('.jpg')]

            for img_file in image_files:
                img_path = os.path.join(breed_path, img_file)
                image_paths.append(img_path)
                labels.append(label_index)

        # Save to cache
        if use_cache:
            print("Saving image paths to cache...")
            with open(cache_file, 'wb') as f:
                pickle.dump({'image_paths': image_paths, 'labels': labels}, f)

        print(f"Found {len(image_paths)} images")
        return image_paths, labels

    def get_transforms(self, train=True):
        """Get data transforms for training or validation"""
        if train and config.AUGMENTATION:
            # Training transforms with augmentation
            transform = transforms.Compose([
                transforms.Resize((self.img_height + 32, self.img_width + 32)),
                transforms.RandomCrop((self.img_height, self.img_width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
            ])
        else:
            # Validation/test transforms (no augmentation)
            transform = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
            ])

        return transform

    def create_dataloaders(self, use_cache=True):
        """Create train, validation, and test DataLoaders"""
        # Load image paths and labels
        image_paths, labels = self.load_image_paths(use_cache=use_cache)

        # Split data
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels,
            test_size=config.TEST_SPLIT,
            random_state=config.RANDOM_SEED,
            stratify=labels
        )

        # Second split: train vs val
        val_size_adjusted = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=config.RANDOM_SEED,
            stratify=y_temp
        )

        print(f"Train set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")

        # Create datasets
        train_dataset = DogBreedDataset(
            X_train, y_train,
            transform=self.get_transforms(train=True)
        )
        val_dataset = DogBreedDataset(
            X_val, y_val,
            transform=self.get_transforms(train=False)
        )
        test_dataset = DogBreedDataset(
            X_test, y_test,
            transform=self.get_transforms(train=False)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        return train_loader, val_loader, test_loader

    def get_breed_name(self, index):
        """Get breed name from index"""
        return self.index_to_label.get(index, "Unknown")

    def save_label_mapping(self):
        """Save label mapping to file"""
        mapping_file = os.path.join(config.MODEL_DIR, 'label_mapping.pkl')
        with open(mapping_file, 'wb') as f:
            pickle.dump({
                'label_to_index': self.label_to_index,
                'index_to_label': self.index_to_label,
                'breed_names': self.breed_names
            }, f)
        print(f"Label mapping saved to {mapping_file}")


if __name__ == "__main__":
    # Test data loader
    loader = DogBreedDataLoader()
    train_loader, val_loader, test_loader = loader.create_dataloaders()
    loader.save_label_mapping()

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
