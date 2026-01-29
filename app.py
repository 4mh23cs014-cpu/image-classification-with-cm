import os
import json
import base64
from io import BytesIO
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model storage
model = None
X_test = None
y_test = None
y_pred = None
classes = None

def train_model():
    """Train image classification model on digits dataset"""
    global model, X_test, y_test, y_pred, classes
    
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    classes = digits.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

def generate_confusion_matrix_image():
    """Generate confusion matrix visualization"""
    if y_test is None or y_pred is None:
        return None
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Image Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """Train the model"""
    try:
        global model, X_test, y_test, y_pred
        
        model, X_test, y_test, y_pred = train_model()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix image
        cm_image = generate_confusion_matrix_image()
        
        return jsonify({
            'success': True,
            'accuracy': round(accuracy, 4),
            'total_samples': len(y_test),
            'correct_predictions': int(np.sum(y_pred == y_test)),
            'cm_image': cm_image,
            'report': report,
            'cm_matrix': cm.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict class for uploaded image"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not trained yet'}), 400
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Process image
        img = Image.open(file).convert('L')  # Convert to grayscale
        img = img.resize((8, 8))  # Resize to 8x8 for digits dataset
        img_array = np.array(img).flatten()
        
        # Make prediction
        prediction = model.predict([img_array])[0]
        probabilities = model.predict_proba([img_array])[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probabilities': {str(i): float(prob) for i, prob in enumerate(probabilities)}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Get current model metrics"""
    try:
        if model is None or y_test is None:
            return jsonify({'success': False, 'error': 'Model not trained yet'}), 400
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return jsonify({
            'success': True,
            'accuracy': round(accuracy, 4),
            'total_samples': len(y_test),
            'correct_predictions': int(np.sum(y_pred == y_test)),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
