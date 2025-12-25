"""
Image preprocessing pipeline for face mask detection
Handles resizing, normalization, and bounding box encoding
"""
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict
import os

class ImagePreprocessor:
    """Preprocess images and annotations for training"""
    
    def __init__(self, target_size: Tuple[int, int] = (416, 416)):
        """
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]))
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1]
        
        Args:
            image: Input image with values in [0, 255]
            
        Returns:
            Normalized image with values in [0, 1]
        """
        return image.astype(np.float32) / 255.0
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load image from path and preprocess
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        return image
    
    def encode_labels(self, labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
        """
        One-hot encode class labels
        
        Args:
            labels: Class indices (N,)
            num_classes: Number of classes
            
        Returns:
            One-hot encoded labels (N, num_classes)
        """
        return tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    def preprocess_annotation(self, annotation: Dict, image_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess single annotation
        
        Args:
            annotation: Annotation dictionary from XML parser
            image_dir: Directory containing images
            
        Returns:
            Tuple of (image, boxes, labels)
        """
        # Load and preprocess image
        image_path = os.path.join(image_dir, annotation['filename'])
        image = self.load_and_preprocess_image(image_path)
        
        # Get boxes and labels
        boxes = annotation['boxes']  # Already normalized to [0, 1]
        labels = self.encode_labels(annotation['labels'])
        
        return image, boxes, labels
    
    def create_dataset_splits(self, annotations: list, 
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            seed: int = 42) -> Dict:
        """
        Split annotations into train/val/test sets
        
        Args:
            annotations: List of annotation dictionaries
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        np.random.seed(seed)
        indices = np.random.permutation(len(annotations))
        
        n_train = int(len(annotations) * train_ratio)
        n_val = int(len(annotations) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        return {
            'train': [annotations[i] for i in train_indices],
            'val': [annotations[i] for i in val_indices],
            'test': [annotations[i] for i in test_indices]
        }


def create_tf_dataset(annotations: list, 
                     image_dir: str,
                     batch_size: int = 16,
                     shuffle: bool = True,
                     preprocessor: ImagePreprocessor = None) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from annotations
    
    Args:
        annotations: List of annotation dictionaries
        image_dir: Directory containing images
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        preprocessor: Image preprocessor instance
        
    Returns:
        tf.data.Dataset
    """
    if preprocessor is None:
        preprocessor = ImagePreprocessor()
    
    def generator():
        for ann in annotations:
            try:
                image, boxes, labels = preprocessor.preprocess_annotation(ann, image_dir)
                # For simplicity, we'll use the first object in each image
                # In a full implementation, you'd handle multiple objects per image
                if len(boxes) > 0:
                    yield image, boxes[0], labels[0]
            except Exception as e:
                print(f"Error processing {ann['filename']}: {e}")
                continue
    
    # Determine output signature
    output_signature = (
        tf.TensorSpec(shape=(416, 416, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(4,), dtype=tf.float32),
        tf.TensorSpec(shape=(3,), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ImagePreprocessor()
    
    print(f"Target size: {preprocessor.target_size}")
    print("\nPreprocessor initialized successfully!")
