import os
import base64
from io import BytesIO
import random
import math
from typing import Tuple

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model storage (simple K-NN implemented with numpy)
model = {
    'X_train': None,  # shape (n_samples, n_features)
    'y_train': None,  # shape (n_samples,)
    'classes': list(range(10)),
    'trained': False
}

# Last evaluation
last_metrics = {
    'accuracy': None,
    'confusion_matrix': None,
    'classification_report': None,
    'total_samples': 0,
    'correct_predictions': 0
}

# --- Dataset synthesis & training utilities ---

def synthesize_digit_image(digit: int, canvas_size: int = 28) -> Image.Image:
    """Render a single synthetic digit image using PIL."""
    img = Image.new('L', (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)

    # Use default font; draw the digit centered and then random transform
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    text = str(digit)
    # center text
    w, h = draw.textsize(text, font=font)
    x = (canvas_size - w) // 2 + random.randint(-2, 2)
    y = (canvas_size - h) // 2 + random.randint(-2, 2)
    draw.text((x, y), text, fill=0, font=font)

    # Random rotation and affine perturbation
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

    # Add some gaussian noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(scale=random.uniform(0, 12), size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_dataset(samples_per_class: int = 40, target_size: Tuple[int, int] = (8, 8)):
    """Create a synthetic digits dataset (X, y) where X are flattened images scaled to target_size."""
    X = []
    y = []
    for digit in range(10):
        for _ in range(samples_per_class):
            img = synthesize_digit_image(digit, canvas_size=28)
            img = img.resize(target_size, resample=Image.BILINEAR)
            arr = np.array(img).astype(np.float32)
            # Invert colors so that ink is high values
            arr = 255.0 - arr
            arr = arr / 255.0  # normalize 0..1
            X.append(arr.flatten())
            y.append(digit)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y


def train_knn(X: np.ndarray, y: np.ndarray):
    """Store training data in model structure."""
    model['X_train'] = X
    model['y_train'] = y
    model['trained'] = True


def knn_predict_one(x: np.ndarray, k: int = 5):
    """Predict a single sample using K-NN and return (prediction, probabilities_array)."""
    if not model['trained']:
        raise RuntimeError('Model not trained')
    X = model['X_train']
    y = model['y_train']
    # Euclidean distances
    dists = np.linalg.norm(X - x, axis=1)
    idx = np.argsort(dists)[:k]
    neighbors = y[idx]
    # Majority vote
    counts = np.bincount(neighbors, minlength=10)
    probs = counts.astype(np.float32) / counts.sum() if counts.sum() > 0 else np.ones(10) / 10
    pred = int(np.argmax(probs))
    return pred, probs.tolist()


# --- Metrics utilities ---

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray):
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report_np(y_true: np.ndarray, y_pred: np.ndarray):
    report = {}
    cm = confusion_matrix_np(y_true, y_pred)
    for i in range(10):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm[i, :].sum())
        report[i] = {
            'precision': float(round(precision, 4)),
            'recall': float(round(recall, 4)),
            'f1-score': float(round(f1, 4)),
            'support': support
        }
    return report


# Generate a simple image for confusion matrix using PIL
def generate_confusion_matrix_image_base64(cm: np.ndarray) -> str:
    n = cm.shape[0]
    cell = 40
    pad = 60
    img_size = (pad + n * cell + 10, pad + n * cell + 10)
    img = Image.new('RGB', img_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    max_val = max(1, int(cm.max()))

    # Draw grid and fill cells
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
            # map to grayscale intensity (higher count -> darker
            intensity = int(255 - (val / max_val) * 200)
            x0 = pad + j * cell
            y0 = pad + i * cell
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            draw.rectangle([x0, y0, x1, y1], fill=(intensity, intensity, 255))
            # draw text centered
            txt = str(val)
            tw, th = draw.textsize(txt)
            tx = x0 + (cell - tw) // 2
            ty = y0 + (cell - th) // 2
            draw.text((tx, ty), txt, fill=(0, 0, 0))

    # Labels
    for i in range(n):
        tx = pad - 10
        ty = pad + i * cell + (cell // 2) - 6
        draw.text((tx - 18, ty), str(i), fill=(0, 0, 0))
        # column labels
        tx2 = pad + i * cell + (cell // 2) - 6
        ty2 = pad - 30
        draw.text((tx2, ty2), str(i), fill=(0, 0, 0))

    # Title
    draw.text((10, 10), 'Confusion Matrix', fill=(0, 0, 0))

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


# --- Flask endpoints ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train():
    try:
        # Build synthetic dataset
        X, y = build_dataset(samples_per_class=40, target_size=(8, 8))
        # Shuffle
        perm = np.random.RandomState(42).permutation(len(y))
        X = X[perm]
        y = y[perm]
        # Split 80/20
        split = int(len(y) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        train_knn(X_train, y_train)

        # Predict on test
        y_pred = np.array([knn_predict_one(x, k=5)[0] for x in X_test])

        # Metrics
        acc = float((y_pred == y_test).mean())
        cm = confusion_matrix_np(y_test, y_pred)
        report = classification_report_np(y_test, y_pred)

        last_metrics['accuracy'] = round(acc, 4)
        last_metrics['confusion_matrix'] = cm.tolist()
        last_metrics['classification_report'] = report
        last_metrics['total_samples'] = int(len(y_test))
        last_metrics['correct_predictions'] = int((y_pred == y_test).sum())

        cm_image = generate_confusion_matrix_image_base64(cm)

        return jsonify({
            'success': True,
            'accuracy': last_metrics['accuracy'],
            'total_samples': last_metrics['total_samples'],
            'correct_predictions': last_metrics['correct_predictions'],
            'cm_image': cm_image,
            'report': report,
            'cm_matrix': cm.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model['trained']:
            return jsonify({'success': False, 'error': 'Model not trained yet'}), 400
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        file = request.files['image']
        img = Image.open(file).convert('L')
        img = img.resize((8, 8), resample=Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        arr = 255.0 - arr
        arr = arr / 255.0
        x = arr.flatten()
        pred, probs = knn_predict_one(x, k=5)
        # Return probabilities as object with numeric keys for compatibility
        probs_obj = {i: float(probs[i]) for i in range(10)}
        return jsonify({'success': True, 'prediction': int(pred), 'probabilities': probs_obj})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    if not model['trained'] or last_metrics['accuracy'] is None:
        return jsonify({'success': False, 'error': 'Model not trained yet'}), 400
    return jsonify({
        'success': True,
        'accuracy': last_metrics['accuracy'],
        'total_samples': last_metrics['total_samples'],
        'correct_predictions': last_metrics['correct_predictions'],
        'confusion_matrix': last_metrics['confusion_matrix'],
        'classification_report': last_metrics['classification_report']
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
