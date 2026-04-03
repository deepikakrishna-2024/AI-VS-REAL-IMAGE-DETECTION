"""
FastAPI Backend for AI vs Real Image Detection (PyTorch)
Provides prediction endpoint for the frontend
Compatible with Python 3.12, 3.13, 3.14
"""

import os
import io
import pickle
import asyncio
import contextlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Global model variable
model = None
class_indices = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup - don't block, load model in background
    asyncio.create_task(load_model_async())
    yield
    # Shutdown (if needed)


async def load_model_async():
    """Load model in background without blocking startup"""
    await asyncio.sleep(0.1)  # Let server start first
    await load_model()


app = FastAPI(
    title="AI vs Real Image Detection API",
    description="ML-powered API to detect AI-generated vs real images",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../ml_model/ai_real_classifier.pth")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "../ml_model/class_indices.pkl")
IMG_SIZE = 224

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global model variable
model = None
class_indices = None


class CNNClassifier(nn.Module):
    """CNN model for binary classification - Fast version (matches train.py)"""
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.backbone = models.resnet18(weights=None)
        
        # Simple classifier head (matches train.py)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_transforms():
    """Get image transforms for inference"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

async def load_model():
    """Load the trained model on startup"""
    global model, class_indices
    
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            
            # Load checkpoint
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            # Create model and load weights
            model = CNNClassifier().to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            print("Model loaded successfully!")
        else:
            print(f"WARNING: Model not found at {MODEL_PATH}")
            print("Please train the model first using ml_model/train.py")
            model = None
        
        # Load class indices
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'rb') as f:
                class_indices = pickle.load(f)
            print(f"Class indices loaded: {class_indices}")
        else:
            # Default mapping
            class_indices = {"FAKE": 0, "REAL": 1}
            print(f"Using default class indices: {class_indices}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None

def preprocess_image(image_bytes: bytes):
    """Preprocess image for model prediction"""
    import time
    start = time.time()
    
    # Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize large images before tensor conversion to save memory
    max_size = 512
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Apply transforms
    transform = get_transforms()
    img_tensor = transform(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    print(f"Image preprocessing took {time.time() - start:.2f}s")
    return img_tensor

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI vs Real Image Detection API",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image is AI-generated or real
    
    Returns:
    - label: "Real" or "AI Generated"
    - confidence: confidence score (0-1)
    - raw_score: raw model output
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_types}"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Read image bytes
        image_bytes = await file.read()
        read_time = time.time()
        print(f"Image read took {read_time - start_time:.2f}s ({len(image_bytes)} bytes)")
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Preprocess image
        img_tensor = preprocess_image(image_bytes)
        preprocess_time = time.time()
        
        # Make prediction
        with torch.no_grad():
            prediction = model(img_tensor)
            raw_score = float(prediction.squeeze().cpu().numpy())
        predict_time = time.time()
        print(f"Model inference took {predict_time - preprocess_time:.2f}s")
        print(f"Total request took {predict_time - start_time:.2f}s")
        
        # Determine label based on class indices
        # Usually: FAKE=0, REAL=1 (or vice versa depending on training)
        if class_indices:
            # Find which class corresponds to index 0 and 1
            idx_to_class = {v: k for k, v in class_indices.items()}
            
            if raw_score < 0.5:
                label_key = idx_to_class.get(0, "FAKE")
                confidence = 1 - raw_score
            else:
                label_key = idx_to_class.get(1, "REAL")
                confidence = raw_score
            
            # Format label
            if label_key.upper() == "FAKE":
                label = "AI Generated"
            else:
                label = "Real"
        else:
            # Fallback if class indices not available
            if raw_score < 0.5:
                label = "AI Generated"
                confidence = 1 - raw_score
            else:
                label = "Real"
                confidence = raw_score
        
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "raw_score": round(raw_score, 4),
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple images"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            img_tensor = preprocess_image(image_bytes)
            
            with torch.no_grad():
                prediction = model(img_tensor)
                raw_score = float(prediction.squeeze().cpu().numpy())
            
            if raw_score < 0.5:
                label = "AI Generated"
                confidence = 1 - raw_score
            else:
                label = "Real"
                confidence = raw_score
            
            results.append({
                "filename": file.filename,
                "label": label,
                "confidence": round(confidence, 4),
                "raw_score": round(raw_score, 4)
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
