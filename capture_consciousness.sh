#!/bin/bash
# DAWN Consciousness Moment Capturer
# Easy launcher script that handles the Python path automatically

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

echo "🌅 DAWN Consciousness Moment Capturer"
echo "=================================="

# Run the consciousness capturer with all arguments passed through
python3 dawn/interfaces/gui/visualization/capture_consciousness_moment.py "$@"
