"""
Face Mask Detection Model Architecture
Transfer learning with MobileNetV2 and dual-head output
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

class MaskDetectorModel:
    """Build face mask detection model with dual heads"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (416, 416, 3),
                 num_classes: int = 3):
        """
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of classes (3: with_mask, without_mask, mask_weared_incorrect)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build(self, trainable_base: bool = False) -> models.Model:
        """
        Build the model architecture
        
        Args:
            trainable_base: Whether to make base model trainable (for fine-tuning)
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # Base model: MobileNetV2 pre-trained on ImageNet
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = trainable_base
        
        # Feature extraction
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Shared dense layer
        shared = layers.Dense(256, activation='relu', name='shared_dense')(x)
        shared = layers.Dropout(0.3)(shared)
        
        # Bounding Box Regression Head
        bbox_head = layers.Dense(128, activation='relu', name='bbox_dense_1')(shared)
        bbox_head = layers.Dropout(0.2)(bbox_head)
        bbox_head = layers.Dense(64, activation='relu', name='bbox_dense_2')(bbox_head)
        bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output')(bbox_head)
        
        # Classification Head
        class_head = layers.Dense(128, activation='relu', name='class_dense_1')(shared)
        class_head = layers.Dropout(0.2)(class_head)
        class_head = layers.Dense(64, activation='relu', name='class_dense_2')(class_head)
        class_output = layers.Dense(self.num_classes, activation='softmax', name='class_output')(class_head)
        
        # Build multi-output model
        model = models.Model(
            inputs=inputs,
            outputs={'bbox': bbox_output, 'class': class_output},
            name='mask_detector'
        )
        
        self.model = model
        return model
    
    def compile(self, 
                learning_rate: float = 0.001,
                bbox_loss_weight: float = 1.0,
                class_loss_weight: float = 1.0):
        """
        Compile the model with custom losses
        
        Args:
            learning_rate: Learning rate for optimizer
            bbox_loss_weight: Weight for bounding box loss
            class_loss_weight: Weight for classification loss
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'bbox': 'mse',  # Mean Squared Error for bbox regression
                'class': 'categorical_crossentropy'  # Categorical CE for classification
            },
            loss_weights={
                'bbox': bbox_loss_weight,
                'class': class_loss_weight
            },
            metrics={
                'bbox': ['mae'],  # Mean Absolute Error
                'class': ['accuracy', tf.keras.metrics.CategoricalAccuracy()]
            }
        )
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model.summary()
    
    def get_model(self) -> models.Model:
        """Get the built model"""
        return self.model


def create_model(input_shape: Tuple[int, int, int] = (416, 416, 3),
                num_classes: int = 3,
                learning_rate: float = 0.001,
                trainable_base: bool = False) -> models.Model:
    """
    Helper function to create and compile model
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
        learning_rate: Learning rate
        trainable_base: Whether to fine-tune base model
        
    Returns:
        Compiled Keras model
    """
    detector = MaskDetectorModel(input_shape=input_shape, num_classes=num_classes)
    model = detector.build(trainable_base=trainable_base)
    detector.compile(learning_rate=learning_rate)
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating face mask detection model...")
    
    model = create_model()
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    
    print("\n" + "="*80)
    print("MODEL OUTPUTS")
    print("="*80)
    print(f"Output names: {list(model.output_names)}")
    
    # Test forward pass
    import numpy as np
    dummy_input = np.random.random((1, 416, 416, 3)).astype(np.float32)
    outputs = model.predict(dummy_input, verbose=0)
    
    print(f"\nBBox output shape: {outputs['bbox'].shape}")
    print(f"Class output shape: {outputs['class'].shape}")
    print(f"\nBBox prediction: {outputs['bbox'][0]}")
    print(f"Class prediction: {outputs['class'][0]}")
    
    print("\n✅ Model created successfully!")
