import tensorflow as tf
import numpy as np
from PIL import Image
import io

class ValidationError(Exception):
    pass

class ScanService:
    def __init__(self):
        self.classes = ["Good", "Heavy Damaged", "Light Damaged"]
        self.descriptions = {
            "Good": "The sidewalk is in good condition. No action needed",
            "Heavy Damaged": "The sidewalk is heavily damaged. Please report this area",
            "Light Damaged": "The sidewalk is lightly damaged. Please report this area",
        }

    def predict(self, image_bytes, model):
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((299, 299))  # Sesuaikan dengan input model
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            input_tensor = tf.expand_dims(img_array, axis=0)

            predictions = model.predict(input_tensor)[0]
            percentages = [float(prob * 100) for prob in predictions]
            max_index = int(np.argmax(percentages))
            label = self.classes[max_index]
            probability = percentages[max_index]
            description = self.descriptions[label]

            return {
                "label": label,
                "probability": probability,
                "description": description
            }

        except Exception as e:
            raise ValidationError("Error processing image: " + str(e))
