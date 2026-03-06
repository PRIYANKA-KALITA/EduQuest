#!/usr/bin/env python3

# gevent monkey patching MUST be first, before any other imports
import gevent.monkey
gevent.monkey.patch_all()

import os
from dotenv import load_dotenv

load_dotenv()

from app import create_app, socketio

# Create the app instance - used by Gunicorn via 'run:app'
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print("=" * 50)
    print("EduQuest SYSTEM STARTING")
    print(f"Listening on http://{host}:{port}")
    print("=" * 50)

    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=False
    )