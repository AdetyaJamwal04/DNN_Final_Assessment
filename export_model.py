"""
Export trained model to SavedModel format for deployment
"""
import tensorflow as tf
import sys
sys.path.append('src')
from models.mask_detector import create_model

# Recreate the model architecture
print("Recreating model architecture...")
model = create_model(
    input_shape=(416, 416, 3),
    num_classes=3,
    learning_rate=0.001,
    trainable_base=False
)

# Load the weights from best checkpoint
print("Loading best weights...")
model.load_weights('models/checkpoints/best_model.h5')

# Export to SavedModel format
print("Exporting to SavedModel format...")
model.export('models/saved_model')

print("\n✅ Model successfully exported to models/saved_model")
print("You can now use it for deployment!")
