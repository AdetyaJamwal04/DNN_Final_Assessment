"""
Test the complete pipeline without training
Tests data loading, model creation, and inference
"""
import os
import sys
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append('src')

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    try:
        from data.xml_parser import VOCParser
        from data.preprocessor import ImagePreprocessor
        from data.augmentation import DataAugmentation
        from models.mask_detector import create_model
        from models.losses import iou_loss
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        from models.mask_detector import create_model
        
        model = create_model(
            input_shape=(416, 416, 3),
            num_classes=3,
            learning_rate=0.001
        )
        
        print("Model created successfully!")
        print(f"Total parameters: {model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        # Test forward pass
        dummy_input = np.random.random((1, 416, 416, 3)).astype(np.float32)
        outputs = model.predict(dummy_input, verbose=0)
        
        print(f"BBox output shape: {outputs['bbox'].shape}")
        print(f"Class output shape: {outputs['class'].shape}")
        
        print("✅ Model creation successful!")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_data_processing():
    """Test data processing pipeline"""
    print("\nTesting data processing...")
    try:
        from data.preprocessor import ImagePreprocessor
        from data.augmentation import DataAugmentation
        
        preprocessor = ImagePreprocessor(target_size=(416, 416))
        augmentation = DataAugmentation(apply_prob=0.5)
        
        # Create dummy data
        dummy_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        dummy_bbox = tf.constant([0.2, 0.3, 0.6, 0.7])
        dummy_label = tf.constant([1.0, 0.0, 0.0])
        
        # Test preprocessing
        resized = preprocessor.resize_image(dummy_image)
        normalized = preprocessor.normalize_image(resized)
        
        print(f"Original shape: {dummy_image.shape}")
        print(f"Resized shape: {resized.shape}")
        print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
        
        # Test augmentation
        aug_image, aug_bbox, aug_label = augmentation.augment(
            tf.constant(normalized), dummy_bbox, dummy_label
        )
        
        print(f"Augmented image shape: {aug_image.shape}")
        print(f"Augmented bbox: {aug_bbox.numpy()}")
        
        print("✅ Data processing successful!")
        return True
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def test_evaluation_metrics():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")
    try:
        from evaluation.metrics import calculate_iou
        
        # Test IoU calculation
        box1 = np.array([0.2, 0.3, 0.6, 0.7])
        box2 = np.array([0.25, 0.35, 0.65, 0.75])
        iou = calculate_iou(box1, box2)
        
        print(f"IoU between similar boxes: {iou:.4f}")
        
        # Test with identical boxes
        box3 = np.array([0.2, 0.3, 0.6, 0.7])
        iou_same = calculate_iou(box1, box3)
        print(f"IoU between identical boxes: {iou_same:.4f}")
        
        assert abs(iou_same - 1.0) < 0.001, "IoU of identical boxes should be 1.0"
        
        print("✅ Evaluation metrics successful!")
        return True
    except Exception as e:
        print(f"❌ Evaluation metrics error: {e}")
        return False

def test_visualization():
    """Test visualization tools"""
    print("\nTesting visualization...")
    try:
        from evaluation.visualize import BBoxVisualizer
        
        visualizer = BBoxVisualizer()
        
        # Create dummy image and bbox
        dummy_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        dummy_bbox = np.array([0.2, 0.3, 0.6, 0.7])
        
        # Draw bbox
        result = visualizer.draw_bbox(dummy_image, dummy_bbox, label=0, confidence=0.95)
        
        print(f"Visualization output shape: {result.shape}")
        print(f"Class names: {visualizer.class_names}")
        
        print("✅ Visualization successful!")
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("TESTING FACE MASK DETECTION SYSTEM")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Processing", test_data_processing),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Visualization", test_visualization)
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
