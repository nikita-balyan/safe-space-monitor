import logging 
import os
from app import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Quick health check on routes
    with app.test_client() as client:
        response = client.get('/')
        logger.info(f"Root route status: {response.status_code}")
        
        response = client.get('/api/current')
        logger.info(f"API /api/current status: {response.status_code}")
    
    # Start Flask server
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on http://0.0.0.0:{port}")
    
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=True,
        threaded=True,
        use_reloader=False
    )
