"""
Model evaluation and visualization script
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import pickle

import config
from data_loader import DogBreedDataLoader


def load_model_and_data():
    """Load trained model and test data"""
    print("Loading model...")
    model_path = os.path.join(config.MODEL_DIR, 'best_model.h5')
    if not os.path.exists(model_path):
        model_path = os.path.join(config.MODEL_DIR, 'dog_breed_model_final.h5')

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")

    print("\nLoading test data...")
    data_loader = DogBreedDataLoader()
    X, y = data_loader.load_data(use_cache=True)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_data(X, y)

    # Load label mapping
    mapping_file = os.path.join(config.MODEL_DIR, 'label_mapping.pkl')
    with open(mapping_file, 'rb') as f:
        label_mapping = pickle.load(f)

    return model, (X_test, y_test), label_mapping


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    # Convert one-hot to class indices
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)

    # Plot
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Dog Breed Classification', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_top_k_accuracy(y_true, y_pred_proba, k_values=[1, 3, 5, 10], save_path=None):
    """Calculate and plot top-k accuracy"""
    y_true_idx = np.argmax(y_true, axis=1)

    top_k_accs = []
    for k in k_values:
        # Get top-k predictions
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]

        # Check if true label is in top-k
        correct = sum([y_true_idx[i] in top_k_preds[i] for i in range(len(y_true_idx))])
        accuracy = correct / len(y_true_idx)
        top_k_accs.append(accuracy)
        print(f"Top-{k} Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Plot
    if save_path:
        plt.figure(figsize=(10, 6))
        plt.bar([f"Top-{k}" for k in k_values], top_k_accs, color='steelblue', alpha=0.8)
        plt.xlabel('Top-K', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Top-K Accuracy Analysis', fontsize=14)
        plt.ylim(0, 1.1)
        for i, v in enumerate(top_k_accs):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top-k accuracy plot saved to: {save_path}")

    return top_k_accs


def visualize_predictions(model, X_test, y_test, label_mapping, num_samples=16):
    """Visualize random predictions"""
    # Random sample
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    # Predict
    y_pred_proba = model.predict(X_sample, verbose=0)
    y_pred_idx = np.argmax(y_pred_proba, axis=1)
    y_true_idx = np.argmax(y_sample, axis=1)

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(X_sample[i])
        axes[i].axis('off')

        true_label = label_mapping['index_to_label'][y_true_idx[i]].split('-')[1].replace('_', ' ')
        pred_label = label_mapping['index_to_label'][y_pred_idx[i]].split('-')[1].replace('_', ' ')
        confidence = y_pred_proba[i][y_pred_idx[i]]

        # Color: green if correct, red if wrong
        color = 'green' if y_true_idx[i] == y_pred_idx[i] else 'red'

        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}",
            fontsize=9,
            color=color,
            weight='bold'
        )

    plt.suptitle('Sample Predictions', fontsize=16, y=1.00)
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'sample_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to: {save_path}")


def analyze_per_class_accuracy(y_true, y_pred, label_mapping):
    """Analyze per-class accuracy"""
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)

    # Get breed names
    breed_names = [
        label_mapping['index_to_label'][i].split('-')[1].replace('_', ' ')
        for i in range(len(label_mapping['index_to_label']))
    ]

    # Classification report
    report = classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=breed_names,
        output_dict=True
    )

    # Convert to DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame(report).transpose()

    # Save to CSV
    csv_path = os.path.join(config.RESULTS_DIR, 'per_class_metrics.csv')
    df.to_csv(csv_path)
    print(f"\nPer-class metrics saved to: {csv_path}")

    # Find best and worst performing breeds
    df_breeds = df[:-3]  # Exclude accuracy, macro avg, weighted avg
    df_breeds = df_breeds.sort_values('f1-score', ascending=False)

    print("\n=== Top 10 Best Performing Breeds ===")
    print(df_breeds.head(10)[['precision', 'recall', 'f1-score', 'support']])

    print("\n=== Top 10 Worst Performing Breeds ===")
    print(df_breeds.tail(10)[['precision', 'recall', 'f1-score', 'support']])

    return df


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Load model and data
    model, (X_test, y_test), label_mapping = load_model_and_data()

    # Predict on test set
    print("\nMaking predictions on test set...")
    y_pred_proba = model.predict(X_test, verbose=1)

    # Basic metrics
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    test_loss, test_acc, test_top5_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Test Top-5 Accuracy: {test_top5_acc:.4f} ({test_top5_acc * 100:.2f}%)")

    # Top-K accuracy
    print("\n" + "=" * 80)
    print("TOP-K ACCURACY")
    print("=" * 80)
    topk_plot_path = os.path.join(config.RESULTS_DIR, 'topk_accuracy.png')
    plot_top_k_accuracy(y_test, y_pred_proba, k_values=[1, 3, 5, 10], save_path=topk_plot_path)

    # Confusion matrix
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    breed_names = [
        label_mapping['breed_names'][i].split('-')[1].replace('_', ' ')
        for i in range(len(label_mapping['breed_names']))
    ]
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred_proba, breed_names, cm_path)

    # Per-class analysis
    print("\n" + "=" * 80)
    print("PER-CLASS ANALYSIS")
    print("=" * 80)
    analyze_per_class_accuracy(y_test, y_pred_proba, label_mapping)

    # Visualize predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    visualize_predictions(model, X_test, y_test, label_mapping, num_samples=16)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED!")
    print("=" * 80)
    print(f"\nAll results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
