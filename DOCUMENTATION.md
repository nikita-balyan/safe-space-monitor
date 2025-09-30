# Safe Space Monitor - Technical Documentation

## Project Overview
A real-time sensory overload monitoring system for neurodivergent individuals with AI-powered interventions.

## Architecture

### Frontend
- **Framework**: Vanilla JavaScript with Chart.js
- **Real-time**: Socket.IO for live updates
- **UI Components**: Bootstrap 5 for responsive design
- **Charts**: Real-time sensor data visualization

### Backend
- **Framework**: Flask with Socket.IO
- **Database**: SQLite with SQLAlchemy
- **AI/ML**: Scikit-learn Random Forest classifier
- **Authentication**: Session-based (extensible to JWT)

### Key Features
1. Real-time sensor monitoring
2. AI-powered overload prediction
3. Interactive calming activities
4. Emergency intervention system
5. Data export and reporting

## API Documentation

### Core Endpoints

#### GET /api/current
Returns current sensor data with predictions
```json
{
  "sensor_data": {
    "noise": 65.2,
    "light": 2300.5,
    "motion": 45.1
  },
  "prediction": 0.34,
  "recommendations": [...]
}