"""
Model quantization for deployment
Convert TensorFlow model to TFLite format with quantization
"""
import tensorflow as tf
import numpy as np
import os
import sys

def convert_to_tflite(model_path: str,
                      output_path: str,
                      quantization: str = 'dynamic'):
    """
    Convert TensorFlow model to TFLite format
    
    Args:
        model_path: Path to saved TensorFlow model
        output_path: Path to save TFLite model
        quantization: Type of quantization ('none', 'dynamic', 'float16', 'int8')
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization
    if quantization == 'dynamic':
        print("Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == 'float16':
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        print("Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # For int8 quantization, you need a representative dataset
        # This is a simplified example
        def representative_dataset():
            for _ in range(100):
                yield [np.random.random((1, 416, 416, 3)).astype(np.float32)]
        converter.representative_dataset = representative_dataset
    else:
        print("No quantization applied")
    
    # Convert
    print("Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model sizes
    if os.path.exists(model_path + '/saved_model.pb'):
        original_size = os.path.getsize(model_path + '/saved_model.pb') / (1024 * 1024)
    else:
        original_size = 0
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\n✅ Conversion complete!")
    print(f"Original model size: {original_size:.2f} MB")
    print(f"TFLite model size: {tflite_size:.2f} MB")
    if original_size > 0:
        print(f"Size reduction: {(1 - tflite_size/original_size)*100:.1f}%")
    print(f"Saved to: {output_path}")


def test_tflite_model(tflite_path: str, test_image: np.ndarray = None):
    """
    Test TFLite model inference
    
    Args:
        tflite_path: Path to TFLite model
        test_image: Test image (optional, will use random if not provided)
    """
    print(f"\nTesting TFLite model: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Number of outputs: {len(output_details)}")
    
    # Create test image if not provided
    if test_image is None:
        test_image = np.random.random((1, 416, 416, 3)).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_image)
    
    import time
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000
    
    # Get outputs
    outputs = {}
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        outputs[output_detail['name']] = output_data
        print(f"Output '{output_detail['name']}' shape: {output_data.shape}")
    
    print(f"\n✅ Inference successful!")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"FPS: {1000/inference_time:.1f}")
    
    return outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert model to TFLite')
    parser.add_argument('--model_path', type=str, default='models/saved_model',
                       help='Path to saved TensorFlow model')
    parser.add_argument('--output', type=str, default='models/tflite/model.tflite',
                       help='Path to save TFLite model')
    parser.add_argument('--quantization', type=str, default='dynamic',
                       choices=['none', 'dynamic', 'float16', 'int8'],
                       help='Quantization type')
    parser.add_argument('--test', action='store_true',
                       help='Test the converted model')
    
    args = parser.parse_args()
    
    # Convert model
    convert_to_tflite(args.model_path, args.output, args.quantization)
    
    # Test if requested
    if args.test:
        test_tflite_model(args.output)
