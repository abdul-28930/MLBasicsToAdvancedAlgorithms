# Diabetes Prediction App

Simple ML web app using FastAPI + Next.js

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Usage
1. Start backend (port 8000)
2. Start frontend (port 3000)
3. Enter 10 feature values
4. Click Predict 