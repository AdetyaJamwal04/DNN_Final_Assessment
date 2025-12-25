"""
Simplified training script for face mask detection
Direct implementation without complex data pipelines
"""
import os
import sys
import tensorflow as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

from data.xml_parser import VOCParser
from models.mask_detector import create_model

def load_data(data_dir, target_size=(416, 416)):
    """Load all data into memory for simplicity"""
    annotations_dir = os.path.join(data_dir, 'annotations')
    images_dir = os.path.join(data_dir,  'images')
    
    parser = VOCParser()
    annotations = parser.parse_directory(annotations_dir)
    
    images = []
    bboxes = []
    labels = []
    
    print(f"Loading {len(annotations)} images...")
    for i, ann in enumerate(annotations):
        if i % 100 == 0:
            print(f"  Loaded {i}/{len(annotations)}")
        
        # Load image
        img_path = os.path.join(images_dir, ann['filename'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        
        # Use first object only
        if len(ann['boxes']) > 0:
            images.append(img)
            bboxes.append(ann['boxes'][0])
            labels.append(ann['labels'][0])
    
    X = np.array(images)
    y_bbox = np.array(bboxes)
    y_class = tf.keras.utils.to_categorical(labels, num_classes=3)
    
    print(f"Loaded {len(X)} samples")
    print(f"X shape: {X.shape}")
    print(f"y_bbox shape: {y_bbox.shape}")
    print(f"y_class shape: {y_class.shape}")
    
    return X, y_bbox, y_class

def main():
    # Load data
    print("Loading dataset...")
    X, y_bbox, y_class = load_data('data/raw')
    
    # Split data
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    X_train, y_bbox_train, y_class_train = X[train_idx], y_bbox[train_idx], y_class[train_idx]
    X_val, y_bbox_val, y_class_val = X[val_idx], y_bbox[val_idx], y_class[val_idx]
    X_test, y_bbox_test, y_class_test = X[test_idx], y_bbox[test_idx], y_class[test_idx]
    
    print(f"\nDataset splits:")
    print(f"Train: {len(X_train)}")
    print(f"Val: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        input_shape=(416, 416, 3),
        num_classes=3,
        learning_rate=0.001,
        trainable_base=False
    )
    
    model.summary()
    
    # Callbacks
    os.makedirs('models/checkpoints', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/checkpoints/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train,
        {'bbox': y_bbox_train, 'class': y_class_train},
        validation_data=(X_val, {'bbox': y_bbox_val, 'class': y_class_val}),
        epochs=30,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = model.evaluate(
        X_test,
        {'bbox': y_bbox_test, 'class': y_class_test},
        verbose=1
    )
    
    print(f"\nTest Results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.4f}")
    
    # Save final model
    os.makedirs('models/saved_model', exist_ok=True)
    model.save('models/saved_model')
    print("\nModel saved to models/saved_model")
    
    print("\n✓ Training completed successfully!")

if __name__ == "__main__":
    main()
