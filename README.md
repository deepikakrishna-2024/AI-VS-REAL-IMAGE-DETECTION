# AI vs Real Image Detection

A web application that uses machine learning to detect whether an image is AI-generated or real. Built with React, FastAPI, and PyTorch.

## Purpose / Use Case

This application helps users **identify AI-generated images** versus **authentic photographs**. In an era where AI image generators (like DALL-E, Midjourney, Stable Diffusion) create increasingly realistic images, this tool provides:

### Who Should Use This?
- **Social Media Users** - Verify if viral images are real before sharing
- **Journalists & Editors** - Fact-check images for news authenticity
- **Content Moderators** - Detect AI-generated content on platforms
- **Educators** - Teach students about AI image detection
- **Researchers** - Study deepfake detection and image forensics
- **General Public** - Verify the authenticity of suspicious images

### Why This Matters
- **Combat Misinformation** - Prevent the spread of fake AI-generated news images
- **Protect Digital Integrity** - Identify synthetic media before it causes harm
- **Research & Education** - Learn how AI detection algorithms work
- **Content Verification** - Ensure images used in professional settings are authentic

## Features

- **Upload Images**: Support for JPG and PNG files (max 5MB)
- **AI Detection**: Uses a trained ResNet-18 CNN model to classify images
- **Fast Inference**: Optimized for quick predictions
- **Modern UI**: Clean, responsive interface with drag-and-drop upload
- **Real-time Feedback**: Shows confidence scores and analysis results

## Tech Stack

### Frontend
- React + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Framer Motion (animations)
- React Router (navigation)

### Backend
- FastAPI (Python)
- PyTorch (deep learning)
- Uvicorn (ASGI server)

### ML Model
- ResNet-18 CNN architecture
- Trained on AI-generated vs real image dataset
- Binary classification (Real vs AI Generated)

## Project Structure

```
├── api/                  # FastAPI backend
│   ├── main.py          # API endpoints
│   └── requirements.txt # Python dependencies
├── ml_model/            # ML training and model files
│   ├── train.py         # Model training script
│   ├── ai_real_classifier.pth  # Trained model weights
│   └── class_indices.pkl       # Class mappings
├── src/                 # React frontend source
│   ├── components/      # UI components
│   ├── pages/          # Page components
│   └── App.tsx         # Main app component
├── scripts/            # Automation scripts
│   └── autostart.cjs   # Development server starter
├── start.bat           # Windows startup script
└── package.json        # Node.js dependencies
```

## Setup Instructions (New Laptop)

### Prerequisites

1. **Node.js** (v18 or higher)
   - Download from https://nodejs.org/
   - Verify: `node --version`

2. **Python** (3.10 or higher)
   - Download from https://python.org/
   - Verify: `python --version`

3. **Git**
   - Download from https://git-scm.com/
   - Verify: `git --version`

### Step 1: Clone the Repository

```bash
git clone https://github.com/deepikakrishna-2024/AI-VS-REAL-IMAGE-DETECTION.git
cd AI-VS-REAL-IMAGE-DETECTION
```

### Step 2: Install Frontend Dependencies

```bash
npm install
```

### Step 3: Install Python Dependencies

```bash
cd api
pip install -r requirements.txt
cd ..
```

### Step 4: Download Model Files

The trained model files are required but not in the repo (they're too large). Either:
- Download from your backup/cloud storage, or
- Train the model locally (see Training section below)

Place these files in `ml_model/`:
- `ai_real_classifier.pth` (trained model weights)
- `class_indices.pkl` (class label mappings)

### Step 5: Start the Application

**Option A: Use the start script (Windows)**
```bash
start.bat
```

**Option B: Manual start**

Terminal 1 (Backend):
```bash
cd api
python main.py
```

Terminal 2 (Frontend):
```bash
npm run dev
```

### Step 6: Open in Browser

Navigate to: http://localhost:8080

## Model Training (Optional)

If you need to retrain the model:

1. Prepare your dataset in `datasets/` folder:
   - `datasets/REAL/` - Real images
   - `datasets/FAKE/` - AI-generated images

2. Run training:
```bash
cd ml_model
python train.py
```

Training takes 10-30 minutes depending on your hardware.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and health check |
| `/health` | GET | Health status |
| `/predict` | POST | Upload image for prediction |

## Development

### Frontend Development
```bash
npm run dev        # Start dev server
npm run build      # Build for production
npm run preview    # Preview production build
```

### Backend Development
```bash
cd api
python main.py     # Start API server
```

## Troubleshooting

### Issue: Site keeps loading
- Check that the ML API is running on port 8000
- Verify model files exist in `ml_model/`
- Check browser console for errors (F12)

### Issue: Model not found
- Download or train the model files
- Place in `ml_model/` directory

### Issue: Python packages not found
```bash
cd api
pip install torch torchvision fastapi uvicorn pillow scikit-learn
```

## Author

**Deepika K**  
Panimalar Engineering College (2nd Year), Electronics and Communication Engineering (ECE)

