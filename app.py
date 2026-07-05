from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

CLASSES = ['burn', 'abrasion', 'laceration', 'bruise', 'normal skin']

DISPLAY_NAMES = {
    'burn':        'Burns',
    'laceration':  'Cuts & Wounds',
    'abrasion':    'Cuts & Wounds',
    'bruise':      'Bleeding',
    'normal skin': 'Normal Skin',
}

def get_severity(raw_class, confidence):
    if raw_class == 'burn':
        return 'severe' if confidence > 0.85 else 'moderate'
    elif raw_class == 'laceration':
        return 'severe' if confidence > 0.80 else 'moderate'
    elif raw_class == 'bruise':
        return 'moderate' if confidence > 0.80 else 'mild'
    else:
        return 'mild'

print("Loading model...")
model = tf.keras.models.load_model('quickaid_model.h5')
print("✅ Model ready")

@app.route('/health', methods=['GET'])
def health():
