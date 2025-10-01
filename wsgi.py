from app import app, socketio
import eventlet

if __name__ == "__main__":
    socketio.run(app, debug=False)