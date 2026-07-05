from flask import Flask, request, jsonify
from flask_cors import CORS
import tf_keras
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
model = tf_keras.models.load_model('quickaid_model.h5')
print("✅ Model ready")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'ok',
        'model':   'QuickAid MobileNetV2 v1.0',
        'classes': CLASSES,
    })

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({ 'error': 'No image provided' }), 400

        # Decode base64 image
        image_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Run inference
        predictions = model.predict(arr, verbose=0)
        scores      = predictions[0].tolist()

        max_index  = int(np.argmax(scores))
        confidence = float(scores[max_index])
        raw_class  = CLASSES[max_index]

        result = {
            'injuryType': DISPLAY_NAMES.get(raw_class, raw_class),
            'rawClass':   raw_class,
            'severity':   get_severity(raw_class, confidence),
            'confidence': confidence,
            'isNormal':   raw_class == 'normal skin',
            'allScores':  [
                {
                    'label':      CLASSES[i],
                    'confidence': round(scores[i], 4),
                }
                for i in range(len(CLASSES))
            ],
        }

        print(f"✅ {raw_class} — {confidence*100:.1f}%")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({ 'error': str(e) }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)