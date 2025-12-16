#!/bin/bash
# BISINDO BATTLE - Quick Start Script
# Linux/Mac Shell Script

echo "============================================================"
echo "   BISINDO BATTLE - Game Edukasi Bahasa Isyarat Indonesia"
echo "============================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

echo "[INFO] Starting game..."
echo ""
echo "Controls:"
echo "  Arrow Keys: Navigate menu"
echo "  Enter: Select"
echo "  Space: Submit gesture"
echo "  D: Toggle debug mode (show landmarks)"
echo "  ESC: Back/Pause"
echo ""
echo "============================================================"
echo ""

# Run game
python game/bisindo_game.py

# Deactivate on exit
deactivate

echo ""
echo "Thanks for playing! :)"
