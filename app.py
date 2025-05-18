import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = tf.keras.models.load_model('my_model_50epochs.keras')

# Load class indices and create a mapping dictionary
class_df = pd.read_excel('class_indices.xlsx')
# Create a dictionary mapping class indices to breed names
class_mapping = dict(zip(class_df['Class Index'], class_df['Class Name']))

def is_cat_breed(breed_name):
    cat_indicators = ['persian', 'siamese', 'maine', 'bengal', 'ragdoll', 'birman', 
                     'abyssinian', 'sphynx', 'manx', 'russian', 'bombay', 'himalayan']
    return any(indicator in breed_name.lower() for indicator in cat_indicators)

def preprocess_image(image):
    # Resize the image to match the input size your model expects (assuming 224x224)
    img = image.resize((224, 224))
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return jsonify({"message": "Pet Breed Classifier API"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the POST request
        file = request.files['file']
        # Open and preprocess the image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        # Get the breed name from our mapping dictionary
        predicted_breed = class_mapping[predicted_class_index]
        # Determine if it's a cat or dog breed
        is_cat = is_cat_breed(predicted_breed)
        animal_type = "I am a Cat" if is_cat else "I am a Dog"
        # Replace underscores with spaces and capitalize words for better display
        predicted_breed = predicted_breed.replace('_', ' ').title()
        confidence = float(predictions[0][predicted_class_index])
        
        return jsonify({
            'breed': predicted_breed,
            'confidence': f'{confidence:.2%}',
            'animal_type': animal_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000))) 