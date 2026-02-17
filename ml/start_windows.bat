@echo off
echo ========================================
echo   CROP PREDICTION SYSTEM LAUNCHER
echo ========================================
echo.

echo Checking if model exists...
if not exist crop_model.pkl (
    echo WARNING: Model not found!
    echo You need to train the model first.
    echo.
    echo Do you want to train the model now? (This takes 1-2 minutes)
    set /p train="Type Y to train, N to exit: "
    if /i "%train%"=="Y" (
        echo.
        echo Training model...
        python train_model.py
        if errorlevel 1 (
            echo.
            echo ERROR: Training failed!
            echo Make sure Crop_recommendation.csv is in this folder.
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo Exiting... Please train the model first with: python train_model.py
        pause
        exit /b 1
    )
)

echo.
echo Starting Crop Prediction System...
echo.
echo The application will open in your browser at:
echo http://localhost:5000
echo.
echo Keep this window open while using the app!
echo Press Ctrl+C to stop the server.
echo.
echo ========================================
echo.

python app.py

pause
