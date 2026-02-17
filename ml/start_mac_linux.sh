#!/bin/bash

echo "========================================"
echo "  CROP PREDICTION SYSTEM LAUNCHER"
echo "========================================"
echo ""

# Check if model exists
if [ ! -f "crop_model.pkl" ]; then
    echo "WARNING: Model not found!"
    echo "You need to train the model first."
    echo ""
    echo "Do you want to train the model now? (This takes 1-2 minutes)"
    read -p "Type Y to train, N to exit: " train
    
    if [ "$train" = "Y" ] || [ "$train" = "y" ]; then
        echo ""
        echo "Training model..."
        python3 train_model.py
        
        if [ $? -ne 0 ]; then
            echo ""
            echo "ERROR: Training failed!"
            echo "Make sure Crop_recommendation.csv is in this folder."
            exit 1
        fi
    else
        echo ""
        echo "Exiting... Please train the model first with: python3 train_model.py"
        exit 1
    fi
fi

echo ""
echo "Starting Crop Prediction System..."
echo ""
echo "The application will open in your browser at:"
echo "http://localhost:5000"
echo ""
echo "Keep this window open while using the app!"
echo "Press Ctrl+C to stop the server."
echo ""
echo "========================================"
echo ""

python3 app.py
