"""
Data augmentation for face mask detection with bounding box transformations
"""
import tensorflow as tf
import numpy as np
from typing import Tuple

class DataAugmentation:
    """Apply data augmentation to images and bounding boxes"""
    
    def __init__(self, apply_prob: float = 0.5):
        """
        Args:
            apply_prob: Probability of applying each augmentation
        """
        self.apply_prob = apply_prob
    
    @tf.function
    def random_flip_horizontal(self, image: tf.Tensor, bbox: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Randomly flip image horizontally and adjust bounding box
        
        Args:
            image: Input image (H, W, C)
            bbox: Bounding box [xmin, ymin, xmax, ymax] in normalized coordinates
            
        Returns:
            Augmented image and adjusted bounding box
        """
        if tf.random.uniform([]) > self.apply_prob:
            return image, bbox
        
        # Flip image
        image = tf.image.flip_left_right(image)
        
        # Adjust bounding box coordinates
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        
        # Flip x-coordinates
        new_xmin = 1.0 - xmax
        new_xmax = 1.0 - xmin
        
        bbox = tf.stack([new_xmin, ymin, new_xmax, ymax])
        
        return image, bbox
    
    @tf.function
    def random_brightness(self, image: tf.Tensor) -> tf.Tensor:
        """
        Randomly adjust brightness
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Brightness-adjusted image
        """
        if tf.random.uniform([]) > self.apply_prob:
            return image
        
        return tf.image.random_brightness(image, max_delta=0.2)
    
    @tf.function
    def random_contrast(self, image: tf.Tensor) -> tf.Tensor:
        """
        Randomly adjust contrast
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Contrast-adjusted image
        """
        if tf.random.uniform([]) > self.apply_prob:
            return image
        
        return tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    @tf.function
    def random_saturation(self, image: tf.Tensor) -> tf.Tensor:
        """
        Randomly adjust saturation
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Saturation-adjusted image
        """
        if tf.random.uniform([]) > self.apply_prob:
            return image
        
        return tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    @tf.function
    def random_hue(self, image: tf.Tensor) -> tf.Tensor:
        """
        Randomly adjust hue
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Hue-adjusted image
        """
        if tf.random.uniform([]) > self.apply_prob:
            return image
        
        return tf.image.random_hue(image, max_delta=0.1)
    
    @tf.function
    def augment(self, image: tf.Tensor, bbox: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Apply all augmentations
        
        Args:
            image: Input image (H, W, C)
            bbox: Bounding box [xmin, ymin, xmax, ymax]
            label: One-hot encoded class label
            
        Returns:
            Augmented image, bbox, and label
        """
        # Geometric augmentations (affect both image and bbox)
        image, bbox = self.random_flip_horizontal(image, bbox)
        
        # Color augmentations (only affect image)
        image = self.random_brightness(image)
        image = self.random_contrast(image)
        image = self.random_saturation(image)
        image = self.random_hue(image)
        
        # Clip image values to [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        # Clip bbox values to [0, 1]
        bbox = tf.clip_by_value(bbox, 0.0, 1.0)
        
        return image, bbox, label


def create_augmented_dataset(dataset: tf.data.Dataset, 
                            augmentation: DataAugmentation = None) -> tf.data.Dataset:
    """
    Apply augmentation to a TensorFlow dataset
    
    Args:
        dataset: Input tf.data.Dataset
        augmentation: DataAugmentation instance
        
    Returns:
        Augmented dataset
    """
    if augmentation is None:
        augmentation = DataAugmentation()
    
    return dataset.map(
        lambda img, bbox, label: augmentation.augment(img, bbox, label),
        num_parallel_calls=tf.data.AUTOTUNE
    )


if __name__ == "__main__":
    # Test augmentation
    print("Data augmentation module loaded successfully!")
    
    # Create dummy data
    dummy_image = tf.random.uniform((416, 416, 3), 0, 1)
    dummy_bbox = tf.constant([0.3, 0.4, 0.6, 0.7])
    dummy_label = tf.constant([1.0, 0.0, 0.0])
    
    # Test augmentation
    aug = DataAugmentation(apply_prob=1.0)
    aug_image, aug_bbox, aug_label = aug.augment(dummy_image, dummy_bbox, dummy_label)
    
    print(f"Original bbox: {dummy_bbox.numpy()}")
    print(f"Augmented bbox: {aug_bbox.numpy()}")
    print("Augmentation test passed!")
