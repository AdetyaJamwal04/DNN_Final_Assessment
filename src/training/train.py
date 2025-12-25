"""
Training script for face mask detection model
"""
import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.xml_parser import VOCParser
from data.preprocessor import ImagePreprocessor, create_tf_dataset
from data.augmentation import create_augmented_dataset, DataAugmentation
from models.mask_detector import create_model

class MaskDetectionTrainer:
    """Training pipeline for face mask detection"""
    
    def __init__(self,
                 data_dir: str,
                 model_save_dir: str = 'models/checkpoints',
                 input_shape: tuple = (416, 416, 3),
                 num_classes: int = 3,
                 batch_size: int = 16,
                 learning_rate: float = 0.001):
        """
        Args:
            data_dir: Directory containing raw data
            model_save_dir: Directory to save model checkpoints
            input_shape: Input image shape
            num_classes: Number of classes
            batch_size: Training batch size
            learning_rate: Initial learning rate
        """
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        os.makedirs(model_save_dir, exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and prepare datasets"""
        print("Loading and preparing data...")
        
        # Parse annotations
        annotations_dir = os.path.join(self.data_dir, 'annotations')
        images_dir = os.path.join(self.data_dir, 'images')
        
        parser = VOCParser()
        annotations = parser.parse_directory(annotations_dir)
        
        print(f"Loaded {len(annotations)} annotations")
        
        # Split data
        preprocessor = ImagePreprocessor(target_size=self.input_shape[:2])
        splits = preprocessor.create_dataset_splits(annotations)
        
        print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        # Create TensorFlow datasets
        train_ds = create_tf_dataset(
            splits['train'], 
            images_dir, 
            batch_size=self.batch_size,
            shuffle=True,
            preprocessor=preprocessor
        )
        
        val_ds = create_tf_dataset(
            splits['val'], 
            images_dir, 
            batch_size=self.batch_size,
            shuffle=False,
            preprocessor=preprocessor
        )
        
        test_ds = create_tf_dataset(
            splits['test'], 
            images_dir, 
            batch_size=self.batch_size,
            shuffle=False,
            preprocessor=preprocessor
        )
        
        # Apply augmentation to training data
        augmentation = DataAugmentation(apply_prob=0.5)
        train_ds = create_augmented_dataset(train_ds, augmentation)
        
        return train_ds, val_ds, test_ds
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.model_save_dir, 
            'best_model_{epoch:02d}_{val_loss:.4f}.h5'
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Learning rate scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def train(self, epochs: int = 50):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
        """
        print("Starting training...")
        
        # Load data
        train_ds, val_ds, test_ds = self.load_and_prepare_data()
        
        # Create model
        print("\nCreating model...")
        model = create_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate,
            trainable_base=False  # Start with frozen base
        )
        
        print("\nModel created successfully!")
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Prepare datasets for multi-output model
        def prepare_for_model(image, bbox, label):
            return image, {'bbox': bbox, 'class': label}
        
        train_ds = train_ds.map(prepare_for_model)
        val_ds = val_ds.map(prepare_for_model)
        
        # Train
        print(f"\nTraining for {epochs} epochs...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.model_save_dir, 'final_model.h5')
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Save in SavedModel format for deployment
        saved_model_path = 'models/saved_model'
        model.save(saved_model_path)
        print(f"SavedModel format saved to: {saved_model_path}")
        
        return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train face mask detection model')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing raw data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MaskDetectionTrainer(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Train
    model, history = trainer.train(epochs=args.epochs)
    
    print("\n✅ Training completed!")
