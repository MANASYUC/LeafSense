# Plant Disease Detection App - Setup Guide

## Prerequisites
- Python 3.8+
- Node.js 16+

## Quick Setup

### 1. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### 3. Run Application
```bash
# Terminal 1: Backend
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 4. Access Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

## Usage
1. Upload a plant image
2. Click "Analyze" 
3. View disease prediction and recommendations

## Troubleshooting
- **Model not found**: Ensure model files are in `ml/models/` directory
- **Import errors**: Activate virtual environment and reinstall dependencies
- **Port conflicts**: Ensure ports 8000 and 5173 are available
