# Replit.md

## Overview

Safe Space Monitor is a real-time sensory regulation assistant designed for neurodiverse children. The application monitors environmental conditions (noise, light, and motion) and provides dual interfaces: a simplified child view with emoji-based status indicators and a detailed caregiver view with numerical data and historical charts. The system uses simulated sensor data to demonstrate overload detection and environmental awareness capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Flask with Jinja2 templating for server-side rendering
- **UI Framework**: Bootstrap 5.3.2 with custom CSS for responsive design
- **Visualization**: Chart.js for real-time data visualization in caregiver view
- **Icons**: Font Awesome 6.4.0 for consistent iconography
- **Real-time Updates**: JavaScript polling (every 2 seconds) for live data updates
- **Dual Interface Design**: 
  - Child View: Emoji status indicators (😊, 😐, 😟) with simple messages
  - Caregiver View: Numerical values, progress bars, and historical charts
- **Responsive Design**: Mobile-first approach with accessible color schemes and large touch targets

### Backend Architecture
- **Framework**: Flask web framework with Python
- **Architecture Pattern**: Modular structure with separated concerns:
  - `app.py`: Main application setup and global configuration
  - `routes.py`: HTTP route handlers and API endpoints
  - `sensor_simulator.py`: Real-time sensor data generation and storage
  - `main.py`: Application entry point with health checks
- **Session Management**: Flask sessions with environment-configurable secret key
- **Proxy Support**: ProxyFix middleware for deployment behind reverse proxies
- **Threading**: Background processes for continuous sensor simulation
- **Machine Learning Integration**: Support for joblib-based ML models for overload prediction

### Data Storage
- **In-Memory Storage**: Python dictionaries and lists for real-time sensor data
- **Data Structure**: Time-series arrays maintaining last 60 seconds of readings
- **Data Retention**: Historical data kept in memory for trend analysis
- **CSV Export**: Optional data persistence to CSV files for analysis
- **No Database Required**: Simplified architecture without external database dependencies

### API Design
- **RESTful Endpoints**:
  - GET `/` - Main dashboard with view mode selection
  - GET `/api/current` - Latest sensor readings (JSON)
  - GET `/api/history` - Historical data for charts (JSON)
  - GET `/api/status` - System health and connectivity status
  - GET `/api/thresholds` - Alert threshold configuration
- **Response Format**: Consistent JSON responses with error handling
- **Real-time Data**: Polling-based updates (2-second intervals) rather than WebSocket connections

### Alert System
- **Multi-level Thresholds**: Warning and danger levels for each sensor type
- **Threshold Configuration**:
  - Noise: Warning at 70dB, Danger at 100dB
  - Light: Warning at 3000 lux, Danger at 8000 lux
  - Motion: Warning at 50 units, Danger at 80 units
- **Visual Feedback**: Color-coded status indicators and emoji changes
- **Accessibility**: High contrast alerts and clear status messaging

## External Dependencies

### Frontend Libraries
- **Bootstrap 5.3.2**: UI framework for responsive design and components
- **Chart.js**: Real-time data visualization and historical trend charts
- **Font Awesome 6.4.0**: Icon library for consistent visual elements

### Backend Libraries
- **Flask**: Web framework for HTTP routing and templating
- **Werkzeug**: WSGI utilities including ProxyFix middleware
- **joblib**: Machine learning model serialization and loading
- **numpy**: Numerical computing for sensor data processing
- **pandas**: Data manipulation for CSV handling and analysis
- **scikit-learn**: Machine learning framework for overload prediction models
- **pathlib**: Modern file system path handling

### Development Tools
- **logging**: Python standard library for application logging
- **threading**: Concurrent sensor simulation and data collection
- **argparse**: Command-line interface for data simulation scripts

### Data Processing
- **CSV files**: Sensor data logging and historical analysis
- **JSON**: API response format and configuration storage
- **Time-series data**: In-memory arrays for real-time monitoring

### Deployment Support
- **Environment Variables**: Configurable session secrets and port settings
- **ProxyFix**: Support for reverse proxy deployments
- **Health Checks**: Built-in endpoint testing and status monitoring