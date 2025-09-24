# Plant Disease Detection App

A React frontend with FastAPI backend for detecting plant diseases from images using machine learning.

## Features

- Upload plant images for disease detection
- Get predictions with confidence scores
- Receive treatment recommendations
- Modern, responsive web interface
- Fast API backend with automatic reloading

## Quick Start

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- Git

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Initialize Git LFS (run once per system):

   ```bash
   git lfs install

   ```

3. Create and activate a virtual environment (recommended):

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the FastAPI server:

   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

   The backend will be available at: http://127.0.0.1:8000

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:

   ```bash
   npm install
   ```

3. Run the React development server:

   ```bash
   npm run dev
   ```

   The frontend will be available at: http://localhost:3000

## Easy Startup Scripts

For convenience, you can use the provided startup scripts:

### Windows

- **Backend**: Double-click `start_backend.bat` or run it from command prompt
- **Frontend**: Double-click `start_frontend.bat` or run it from command prompt

### Linux/Mac

- **Backend**: `./start_backend.sh`
- **Frontend**: `./start_frontend.sh`

## Usage

1. Start both the backend and frontend servers
2. Open your browser to http://localhost:3000
3. Upload a plant image using the interface
4. View the disease prediction results and treatment recommendations

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /api/analyze` - Analyze plant image for diseases

## Project Structure

```
plant_detect_react/
├── backend/
│   ├── api/
│   │   └── main.py          # FastAPI application
│   ├── schemas.py           # Pydantic models
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── App.tsx         # Main app component
│   │   └── main.tsx        # Entry point
│   ├── package.json        # Node.js dependencies
│   └── vite.config.ts      # Vite configuration
├── ml/
│   └── models/             # Trained ML models
├── data/                   # Training and test data
├── start_backend.bat       # Windows backend startup
├── start_backend.sh        # Linux/Mac backend startup
├── start_frontend.bat      # Windows frontend startup
├── start_frontend.sh       # Linux/Mac frontend startup
└── model_load.py          # Model loading and prediction logic
```

## Development

### Backend Development

- The FastAPI server runs with auto-reload enabled
- API documentation is available at http://127.0.0.1:8000/docs
- CORS is configured to allow requests from the frontend

### Frontend Development

- Built with React + TypeScript + Vite
- Hot module replacement enabled
- ESLint configured for code quality

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in the startup command

   - Backend: `uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload`
   - Frontend: Update `vite.config.ts` or use `npm run dev -- --port 3001`

2. **Python dependencies issues**: Make sure you're using the correct Python version and virtual environment

3. **Node.js dependencies issues**: Try deleting `node_modules` and running `npm install` again

4. **CORS errors**: Ensure the backend is running and the frontend URL is allowed in the CORS settings

### Getting Help

- Check the console output for error messages
- Ensure all dependencies are properly installed
- Verify that both servers are running on the correct ports

## License

This project is for educational and research purposes.
