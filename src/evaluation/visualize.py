"""
Visualization tools for face mask detection
Draw bounding boxes, display predictions, create evaluation reports
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import os

class BBoxVisualizer:
    """Visualize bounding boxes and predictions"""
    
    def __init__(self):
        self.class_colors = {
            0: (0, 255, 0),      # with_mask: Green
            1: (255, 0, 0),      # without_mask: Red
            2: (255, 255, 0)     # mask_weared_incorrect: Yellow
        }
        self.class_names = {
            0: 'with_mask',
            1: 'without_mask',
            2: 'mask_weared_incorrect'
        }
    
    def draw_bbox(self, image: np.ndarray, 
                  bbox: np.ndarray, 
                  label: int,
                  confidence: float = None,
                  thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on image
        
        Args:
            image: Input image (H, W, C) in RGB
            bbox: Bounding box [xmin, ymin, xmax, ymax] in normalized coordinates
            label: Class label (0, 1, or 2)
            confidence: Confidence score (optional)
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding box
        """
        image = image.copy()
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        xmin = int(bbox[0] * w)
        ymin = int(bbox[1] * h)
        xmax = int(bbox[2] * w)
        ymax = int(bbox[3] * h)
        
        # Get color and class name
        color = self.class_colors.get(label, (255, 255, 255))
        class_name = self.class_names.get(label, 'unknown')
        
        # Draw rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        
        # Prepare label text
        if confidence is not None:
            label_text = f'{class_name}: {confidence:.2f}'
        else:
            label_text = class_name
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(image, (xmin, ymin - text_height - 10), 
                     (xmin + text_width, ymin), color, -1)
        
        # Draw label text
        cv2.putText(image, label_text, (xmin, ymin - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image
    
    def visualize_prediction(self, image: np.ndarray,
                           true_bbox: np.ndarray,
                           true_label: int,
                           pred_bbox: np.ndarray,
                           pred_label: int,
                           pred_confidence: float = None) -> np.ndarray:
        """
        Visualize ground truth and prediction side by side
        
        Args:
            image: Input image
            true_bbox: Ground truth bounding box
            true_label: Ground truth label
            pred_bbox: Predicted bounding box
            pred_label: Predicted label
            pred_confidence: Prediction confidence
            
        Returns:
            Combined visualization image
        """
        # Draw ground truth
        img_true = self.draw_bbox(image, true_bbox, true_label)
        
        # Draw prediction
        img_pred = self.draw_bbox(image, pred_bbox, pred_label, pred_confidence)
        
        # Combine side by side
        combined = np.hstack([img_true, img_pred])
        
        return combined
    
    def save_predictions(self, images: List[np.ndarray],
                        true_boxes: List[np.ndarray],
                        true_labels: List[int],
                        pred_boxes: List[np.ndarray],
                        pred_labels: List[int],
                        pred_confidences: List[float],
                        save_dir: str,
                        num_images: int = 10):
        """
        Save prediction visualizations
        
        Args:
            images: List of images
            true_boxes: List of ground truth boxes
            true_labels: List of ground truth labels
            pred_boxes: List of predicted boxes
            pred_labels: List of predicted labels
            pred_confidences: List of prediction confidences
            save_dir: Directory to save visualizations
            num_images: Number of images to save
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(min(num_images, len(images))):
            combined = self.visualize_prediction(
                images[i], true_boxes[i], true_labels[i],
                pred_boxes[i], pred_labels[i], pred_confidences[i]
            )
            
            # Convert RGB to BGR for saving with cv2
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(save_dir, f'prediction_{i:03d}.png')
            cv2.imwrite(save_path, combined_bgr)
        
        print(f"Saved {min(num_images, len(images))} prediction visualizations to {save_dir}")


def plot_confusion_matrix(conf_matrix: np.ndarray,
                          class_names: List[str],
                          save_path: str = None):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics(metrics_dict: dict, save_path: str = None):
    """
    Plot evaluation metrics
    
    Args:
        metrics_dict: Dictionary with evaluation metrics
        save_path: Path to save figure (optional)
    """
    class_names = metrics_dict['class_names']
    report = metrics_dict['classification_report']
    
    # Extract precision, recall, f1-score for each class
    precisions = [report[name]['precision'] for name in class_names]
    recalls = [report[name]['recall'] for name in class_names]
    f1_scores = [report[name]['f1-score'] for name in class_names]
    
    # Create bar plot
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    
    # Test visualizer
    visualizer = BBoxVisualizer()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    dummy_bbox = np.array([0.2, 0.3, 0.6, 0.7])
    
    # Draw bbox
    result = visualizer.draw_bbox(dummy_image, dummy_bbox, label=0, confidence=0.95)
    
    print("\n✅ Visualization functions working correctly!")
