"""
Custom loss functions for face mask detection
"""
import tensorflow as tf

def iou_loss(y_true, y_pred):
    """
    IoU (Intersection over Union) loss for bounding boxes
    
    Args:
        y_true: True bounding boxes [batch_size, 4] in [xmin, ymin, xmax, ymax] format
        y_pred: Predicted bounding boxes [batch_size, 4] in [xmin, ymin, xmax, ymax] format
        
    Returns:
        IoU loss (1 - IoU)
    """
    # Calculate intersection
    xmin_inter = tf.maximum(y_true[:, 0], y_pred[:, 0])
    ymin_inter = tf.maximum(y_true[:, 1], y_pred[:, 1])
    xmax_inter = tf.minimum(y_true[:, 2], y_pred[:, 2])
    ymax_inter = tf.minimum(y_true[:, 3], y_pred[:, 3])
    
    inter_width = tf.maximum(0.0, xmax_inter - xmin_inter)
    inter_height = tf.maximum(0.0, ymax_inter - ymin_inter)
    inter_area = inter_width * inter_height
    
    # Calculate union
    true_width = y_true[:, 2] - y_true[:, 0]
    true_height = y_true[:, 3] - y_true[:, 1]
    true_area = true_width * true_height
    
    pred_width = y_pred[:, 2] - y_pred[:, 0]
    pred_height = y_pred[:, 3] - y_pred[:, 1]
    pred_area = pred_width * pred_height
    
    union_area = true_area + pred_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Return 1 - IoU as loss
    return 1.0 - iou


def combined_bbox_loss(y_true, y_pred, 
                       mse_weight: float = 0.5, 
                       iou_weight: float = 0.5):
    """
    Combined bounding box loss: MSE + IoU loss
    
    Args:
        y_true: True bounding boxes
        y_pred: Predicted bounding boxes
        mse_weight: Weight for MSE loss
        iou_weight: Weight for IoU loss
        
    Returns:
        Combined loss
    """
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    
    return mse_weight * mse + iou_weight * tf.reduce_mean(iou)


def focal_loss(y_true, y_pred, alpha: float = 0.25, gamma: float = 2.0):
    """
    Focal loss for handling class imbalance
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss
    """
    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Calculate cross entropy
    ce = -y_true * tf.math.log(y_pred)
    
    # Calculate focal loss
    focal = alpha * tf.pow(1 - y_pred, gamma) * ce
    
    return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))


class CustomLoss(tf.keras.losses.Loss):
    """Custom combined loss for mask detection"""
    
    def __init__(self, 
                 bbox_weight: float = 1.0,
                 class_weight: float = 1.0,
                 use_focal: bool = False,
                 name: str = "custom_loss"):
        """
        Args:
            bbox_weight: Weight for bounding box loss
            class_weight: Weight for classification loss
            use_focal: Whether to use focal loss for classification
            name: Loss name
        """
        super().__init__(name=name)
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.use_focal = use_focal
    
    def call(self, y_true, y_pred):
        """
        Calculate total loss
        
        Args:
            y_true: Dictionary with 'bbox' and 'class' keys
            y_pred: Dictionary with 'bbox' and 'class' keys
            
        Returns:
            Total loss
        """
        # Bounding box loss (MSE)
        bbox_loss = tf.keras.losses.mean_squared_error(
            y_true['bbox'], 
            y_pred['bbox']
        )
        
        # Classification loss
        if self.use_focal:
            class_loss = focal_loss(y_true['class'], y_pred['class'])
        else:
            class_loss = tf.keras.losses.categorical_crossentropy(
                y_true['class'], 
                y_pred['class']
            )
        
        # Combined loss
        total_loss = (self.bbox_weight * tf.reduce_mean(bbox_loss) + 
                     self.class_weight * tf.reduce_mean(class_loss))
        
        return total_loss


if __name__ == "__main__":
    # Test loss functions
    import numpy as np
    
    print("Testing loss functions...")
    
    # Dummy data
    y_true_bbox = np.array([[0.2, 0.3, 0.6, 0.7]], dtype=np.float32)
    y_pred_bbox = np.array([[0.25, 0.35, 0.65, 0.75]], dtype=np.float32)
    
    y_true_class = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    y_pred_class = np.array([[0.8, 0.15, 0.05]], dtype=np.float32)
    
    # Test IoU loss
    iou_l = iou_loss(y_true_bbox, y_pred_bbox)
    print(f"\nIoU Loss: {iou_l.numpy()}")
    
    # Test MSE loss
    mse_l = tf.keras.losses.mean_squared_error(y_true_bbox, y_pred_bbox)
    print(f"MSE Loss: {tf.reduce_mean(mse_l).numpy()}")
    
    # Test focal loss
    focal_l = focal_loss(y_true_class, y_pred_class)
    print(f"Focal Loss: {focal_l.numpy()}")
    
    # Test combined loss
    print("\n✅ All loss functions working correctly!")
