# Image Classification with Confusion Matrix

A Flask web application for image classification (handwritten digits recognition) with comprehensive confusion matrix visualization and analysis.

## Features

- **Model Training**: Train a Random Forest classifier on the digits dataset
- **Confusion Matrix**: Visual heatmap showing classification performance
- **Classification Report**: Detailed metrics for each digit class
- **Image Prediction**: Upload custom images for prediction
- **Responsive UI**: Beautiful, modern web interface
- **Render-Ready**: Fully configured for cloud deployment

## Project Structure

```
image-classification-with-cm/
├── app.py                 # Flask application with ML models
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment configuration
├── README.md             # This file
└── templates/
    └── index.html        # Web interface
```

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd image-classification-with-cm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Training the Model
1. Click the **"Train Model"** button on the home page
2. The application will train a Random Forest classifier on the digits dataset (0-9)
3. View the results including:
   - Overall accuracy
   - Confusion matrix heatmap
   - Classification report

### Making Predictions
1. Upload a handwritten digit image (8x8 or 28x28 pixels, grayscale preferred)
2. Click **"Predict"** button
3. View the predicted digit and confidence scores for all classes

## API Endpoints

### `POST /train`
Trains the image classification model.

**Response:**
```json
{
  "success": true,
  "accuracy": 0.9722,
  "total_samples": 360,
  "correct_predictions": 350,
  "cm_image": "base64_encoded_image",
  "report": { ... },
  "cm_matrix": [ ... ]
}
```

### `POST /predict`
Makes a prediction on an uploaded image.

**Parameters:**
- `image` (file): Image file to classify

**Response:**
```json
{
  "success": true,
  "prediction": 5,
  "probabilities": {
    "0": 0.01,
    "1": 0.02,
    ...
    "5": 0.95,
    ...
  }
}
```

### `GET /metrics`
Retrieves current model metrics.

**Response:**
```json
{
  "success": true,
  "accuracy": 0.9722,
  "total_samples": 360,
  "correct_predictions": 350,
  "confusion_matrix": [ ... ],
  "classification_report": { ... }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Deployment to Render

### Prerequisites
- Render account (free tier available)
- GitHub repository with your code

### Deployment Steps

1. **Push code to GitHub:**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Create New Service on Render:**
   - Go to [render.com](https://render.com)
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Choose the repository and branch

3. **Configure Service:**
   - **Name**: `image-classification` (or your preferred name)
   - **Environment**: Python 3
   - **Build Command**: `bash build.sh`
   - **Start Command**: `gunicorn --workers 2 --timeout 60 --bind 0.0.0.0:$PORT app:app`
   - *Tip:* The repo includes a `render.yaml` that sets these automatically when present.
   - **Environments** (optional):
     - Add any environment variables if needed
   - **Instance Type**: Free tier is sufficient for this demo

4. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy
   - Monitor logs for any issues
   - Your app will be available at `https://your-service-name.onrender.com`

### Environment Variables (Optional)
If needed, set these in Render dashboard:
```
PORT=5000
FLASK_ENV=production
```

## Model Details

### Algorithm
- **Classifier**: Random Forest (100 trees)
- **Dataset**: Scikit-learn digits dataset (1,797 samples, 64 features)
- **Train-Test Split**: 80-20 split with stratification
- **Features**: 8x8 pixel images flattened to 64 features

### Performance
Typical metrics on digits dataset:
- **Accuracy**: ~97-98%
- **Precision**: 0.96-0.99 per class
- **Recall**: 0.95-0.98 per class

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 2.3.3 | Web framework |
| scikit-learn | 1.3.2 | Machine learning |
| matplotlib | 3.7.2 | Plotting |
| seaborn | 0.12.2 | Statistical visualization |
| numpy | 1.24.3 | Numerical computing |
| Pillow | 10.0.0 | Image processing |
| gunicorn | 21.2.0 | WSGI server (production) |

## Troubleshooting

### Model training is slow
- This is normal for the first run
- Model is cached in memory after training
- Subsequent requests use the cached model

### Image prediction not working
- Ensure image is valid (JPEG, PNG, GIF)
- Grayscale images work best
- Image should contain a handwritten digit

### Port already in use
- Change port in `app.py` or set `PORT` environment variable
- Kill process using port 5000: `lsof -i :5000` (macOS/Linux)

### Render deployment fails
- Check logs in Render dashboard
- Ensure `requirements.txt` is in root directory
- Verify `Procfile` exists with correct content
- Check Python version compatibility

## Limitations

- Single model instance (no persistence between restarts)
- Limited to digits dataset (0-9)
- Image predictions best for 8x8 or 28x28 pixel images
- Free tier Render may have cold start delays

## Future Enhancements

- Add database for model persistence
- Support for multiple datasets (CIFAR-10, ImageNet, etc.)
- Model versioning and A/B testing
- Advanced image preprocessing
- Real-time training progress updates
- Model comparison tools

## License

MIT License - Feel free to use this project for educational and commercial purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review application logs
3. Open an issue on GitHub

## Author

Created as an educational project for image classification and ML deployment.
