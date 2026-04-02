#!/bin/bash
# Start Flask API via Gunicorn
gunicorn server:app --bind 0.0.0.0:$PORT --workers 1