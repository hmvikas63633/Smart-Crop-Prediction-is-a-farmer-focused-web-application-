Run Locally (Step by Step)
Open terminal in project folder:
c:\Users\student\Desktop\ml

Create virtual environment:
python -m venv venv

Activate virtual environment (Windows):
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Check required files:
Make sure crop_model.pkl is present in the project root.

Start the Flask server:
python app.py

Open in browser:
http://localhost:5000

Useful routes:

Login: http://localhost:5000/login
Register: http://localhost:5000/register
Predict: http://localhost:5000/predict
Grow Plan: http://localhost:5000/grow-plan
Stop server:
Press Ctrl + C in terminal.
