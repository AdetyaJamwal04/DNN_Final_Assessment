"""
Model loading utilities for Keras 3 compatibility
"""
import tensorflow as tf
import os

def load_model_keras3(model_path):
    """
    Load model with Keras 3 compatibility
    
    Args:
        model_path: Path to the model (SavedModel format or .h5)
        
    Returns:
        Loaded model
    """
    if os.path.isdir(model_path):
        # SavedModel format
        print(f"Loading SavedModel from {model_path}...")
        # For Keras 3, use TFSMLayer for SavedModel
        return tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    elif model_path.endswith('.h5'):
        # H5 format
        print(f"Loading H5 model from {model_path}...")
        return tf.keras.models.load_model(model_path)
    elif model_path.endswith('.keras'):
        # Keras V3 format
        print(f"Loading Keras V3 model from {model_path}...")
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

def predict_with_model(model, image):
    """
    Run prediction handling both model types
    
    Args:
        model: Loaded model (can be TFSMLayer or regular model)
        image: Input image tensor (batch, height, width, channels)
        
    Returns:
        Dictionary with 'bbox' and 'class' predictions
    """
    if isinstance(model, tf.keras.layers.TFSMLayer):
        # TFSMLayer returns a dictionary
        output = model(image)
        # Extract outputs from the dictionary
        if isinstance(output, dict):
            return {
                'bbox': output.get('bbox_output', output.get('bbox', None)),
                'class': output.get('class_output', output.get('class', None))
            }
        return output
    else:
        # Regular model
        return model.predict(image, verbose=0)
