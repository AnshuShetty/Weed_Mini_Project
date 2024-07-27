from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('Dataset1_model.h5')

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('weed_detection.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image, target_size=(150, 150))
        prediction = model.predict(processed_image).tolist()

        # Assuming binary classification: 0 = No Weed, 1 = Weed
        label = 'Weed' if prediction[0][0] > 0.5 else 'No Weed'
        confidence = prediction[0][0] if label == 'Weed' else 1 - prediction[0][0]
        
        return jsonify({'label': label, 'confidence': float(confidence)})
    
    return 'Predict route: Upload an image using POST request'

if __name__ == '__main__':
    app.run(debug=True)
