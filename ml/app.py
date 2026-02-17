from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import joblib
from flask_cors import CORS
import os
import requests
import glob
import re

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import kagglehub
except Exception:
    kagglehub = None

# Create Flask app
app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# External API configuration
# Supports either API_KEY or DATA_GOV_API_KEY environment variable names.
API_KEY = os.getenv("API_KEY") or os.getenv("DATA_GOV_API_KEY") or "YOUR_DATA_GOV_API_KEY"
AGMARKNET_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
KAGGLE_DATASET_REF = "santoshd3/crop-price-prediction"

# Load trained ML model
model = joblib.load("crop_model.pkl")

if not hasattr(model, "predict"):
    raise TypeError(
        "Invalid model artifact: crop_model.pkl does not contain a trained model with a predict() method. "
        "Run train_model.py to regenerate the model file."
    )

CROP_PRICE_INDEX = {}
CROP_PRICE_META = {
    "ready": False,
    "dataset_path": None,
    "csv_file": None,
    "reason": None
}
MODEL_CROPS = set()
if hasattr(model, "classes_"):
    MODEL_CROPS = {str(c).strip().lower() for c in model.classes_}

CROP_ALIASES = {
    "paddy": "rice",
    "cotton lint": "cotton",
    "cotton(lint)": "cotton",
    "kidney beans": "kidneybeans",
    "kidney bean": "kidneybeans",
    "mung bean": "mungbean",
    "moth bean": "mothbeans",
    "black gram": "blackgram",
    "pigeon peas": "pigeonpeas",
}


def _normalize_crop_name(value):
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(value).strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _canonicalize_crop_name(value):
    name = _normalize_crop_name(value)
    if not name:
        return ""
    if name in CROP_ALIASES:
        name = CROP_ALIASES[name]
    if name in MODEL_CROPS:
        return name

    if name.endswith("s") and name[:-1] in MODEL_CROPS:
        return name[:-1]

    compact = name.replace(" ", "")
    if compact in MODEL_CROPS:
        return compact

    for crop in MODEL_CROPS:
        if crop in name or name in crop:
            return crop
    return compact


def _find_column(df, candidates, contains_hints=None):
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    if contains_hints:
        for lower_name, original in lower_map.items():
            if any(hint in lower_name for hint in contains_hints):
                return original
    return None


def _build_price_index(df, crop_col, price_col):
    work = df[[crop_col, price_col]].copy()
    work[crop_col] = work[crop_col].astype(str).map(_canonicalize_crop_name)
    work[price_col] = pd.to_numeric(
        work[price_col].astype(str).str.replace(",", "", regex=False).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )
    work = work.dropna(subset=[crop_col, price_col])
    if work.empty:
        return {}

    grouped = work.groupby(crop_col)[price_col].agg(["median", "count"]).reset_index()
    index = {}
    for _, row in grouped.iterrows():
        crop_name = str(row[crop_col])
        if not crop_name:
            continue
        index[crop_name] = {
            "predicted_price": round(float(row["median"]), 2),
            "sample_size": int(row["count"])
        }
    return index


def _select_crop_column(df):
    best_col = None
    best_score = -1
    for col in df.columns:
        series = df[col]
        if series.dtype.kind not in ("O", "U", "S"):
            continue
        sample = series.dropna().astype(str).head(1000)
        if sample.empty:
            continue
        normalized = sample.map(_canonicalize_crop_name)
        unique_names = {name for name in normalized if name}
        overlap = len(unique_names.intersection(MODEL_CROPS)) if MODEL_CROPS else len(unique_names)
        score = overlap
        lower_col = str(col).lower()
        if "crop" in lower_col or "commodity" in lower_col:
            score += 3
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def _select_price_column(df):
    best_col = None
    best_score = -1.0
    for col in df.columns:
        raw = df[col].astype(str)
        numeric = pd.to_numeric(
            raw.str.replace(",", "", regex=False).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce"
        )
        valid_ratio = float(numeric.notna().mean())
        if valid_ratio <= 0:
            continue
        score = valid_ratio
        lower_col = str(col).lower()
        if "price" in lower_col:
            score += 1.0
        if "modal" in lower_col or "avg" in lower_col or "average" in lower_col:
            score += 0.5
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def load_kaggle_crop_price_data():
    global CROP_PRICE_INDEX, CROP_PRICE_META

    if pd is None:
        CROP_PRICE_META["reason"] = "pandas is not installed"
        return
    if kagglehub is None:
        CROP_PRICE_META["reason"] = "kagglehub is not installed"
        return

    try:
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_REF)
        CROP_PRICE_META["dataset_path"] = dataset_path
    except Exception as e:
        CROP_PRICE_META["reason"] = f"kaggle download failed: {str(e)}"
        return

    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        CROP_PRICE_META["reason"] = "no CSV files found in dataset"
        return

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue

        crop_col = _select_crop_column(df)
        price_col = _select_price_column(df)
        if not crop_col or not price_col:
            continue

        index = _build_price_index(df, crop_col, price_col)
        if index:
            CROP_PRICE_INDEX = index
            CROP_PRICE_META["ready"] = True
            CROP_PRICE_META["csv_file"] = csv_file
            CROP_PRICE_META["reason"] = None
            return

    CROP_PRICE_META["reason"] = "could not identify valid crop/price columns"


load_kaggle_crop_price_data()

# Home route now opens login first
@app.route("/")
def home():
    login_path = os.path.join(BASE_DIR, "login.html")
    if os.path.exists(login_path):
        return send_from_directory(BASE_DIR, "login.html")
    return "Login page not found", 404


@app.route("/predict")
@app.route("/index")
def prediction_page():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(BASE_DIR, "index.html")
    return "Prediction page not found", 404


@app.route("/login")
@app.route("/logon")
def login_page():
    login_path = os.path.join(BASE_DIR, "login.html")
    if os.path.exists(login_path):
        return send_from_directory(BASE_DIR, "login.html")
    return "Login page not found", 404


@app.route("/register")
def register_page():
    register_path = os.path.join(BASE_DIR, "register.html")
    if os.path.exists(register_path):
        return send_from_directory(BASE_DIR, "register.html")
    return "Register page not found", 404


@app.route("/grow-plan")
def grow_plan_page():
    plan_path = os.path.join(BASE_DIR, "grow-plan.html")
    if os.path.exists(plan_path):
        return send_from_directory(BASE_DIR, "grow-plan.html")
    return "Grow plan page not found", 404


@app.route("/api/crops", methods=["GET"])
def list_crops():
    try:
        crops = []
        if hasattr(model, "classes_"):
            crops = sorted([str(item) for item in model.classes_], key=lambda x: x.lower())
        return jsonify({
            "success": True,
            "crops": crops
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to load crop list: {str(e)}"
        }), 500


@app.route("/api/crop-price-predict", methods=["GET"])
def predict_crop_price():
    try:
        crop_input = request.args.get("crop", "")
        crop_name = _canonicalize_crop_name(crop_input)
        if not crop_name:
            return jsonify({
                "success": False,
                "error": "Query parameter 'crop' is required."
            }), 400

        if not CROP_PRICE_META["ready"]:
            return jsonify({
                "success": False,
                "error": "Crop price dataset not ready.",
                "reason": CROP_PRICE_META["reason"]
            }), 503

        match_name = crop_name
        stats = CROP_PRICE_INDEX.get(match_name)
        if stats is None:
            return jsonify({
                "success": False,
                "error": f"No price data found for crop '{crop_input}'."
            }), 404

        return jsonify({
            "success": True,
            "crop": match_name,
            "predicted_price": stats["predicted_price"],
            "sample_size": stats["sample_size"],
            "source": {
                "dataset": KAGGLE_DATASET_REF,
                "csv_file": CROP_PRICE_META["csv_file"]
            }
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to predict crop price: {str(e)}"
        }), 500

# Prediction API
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        nitrogen = float(data["nitrogen"])
        phosphorus = float(data["phosphorus"])
        potassium = float(data["potassium"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        # Convert to model input format
        features = np.array([[nitrogen, phosphorus, potassium,
                              temperature, humidity, ph, rainfall]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Send response
        return jsonify({
            "success": True,
            "recommendations": [
                {
                    "crop": prediction,
                    "confidence": 96,
                    "emoji": "",
                    "description": "Best crop based on soil & climate conditions"
                }
            ]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/api/mandi-prices", methods=["GET"])
def mandi_prices():
    try:
        if not API_KEY or API_KEY == "YOUR_DATA_GOV_API_KEY":
            return jsonify({
                "success": False,
                "error": "API key is missing. Set API_KEY (or DATA_GOV_API_KEY) environment variable."
            }), 500

        limit = request.args.get("limit", 20, type=int)
        if limit is None or limit <= 0:
            return jsonify({
                "success": False,
                "error": "Query parameter 'limit' must be a positive integer."
            }), 400

        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": min(limit, 100)
        }

        # Optional filters pass-through to Agmarknet API.
        commodity = request.args.get("commodity")
        state = request.args.get("state")
        market = request.args.get("market")

        if commodity:
            params["filters[commodity]"] = commodity
        if state:
            params["filters[state]"] = state
        if market:
            params["filters[market]"] = market

        response = requests.get(AGMARKNET_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("records", [])

        prices = []
        for record in records:
            prices.append({
                "crop": record.get("commodity"),
                "market": record.get("market"),
                "state": record.get("state"),
                "min_price": record.get("min_price"),
                "max_price": record.get("max_price"),
                "date": (
                    record.get("arrival_date")
                    or record.get("price_date")
                    or record.get("reported_date")
                )
            })

        return jsonify({
            "success": True,
            "prices": prices
        }), 200

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "Mandi price service timed out. Please try again."
        }), 504
    except requests.exceptions.RequestException as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        return jsonify({
            "success": False,
            "error": f"Failed to fetch mandi prices: {str(e)}",
            "upstream_status": status_code
        }), 502
    except ValueError:
        return jsonify({
            "success": False,
            "error": "Received invalid JSON from mandi price service."
        }), 502
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500


# Run server
if __name__ == "__main__":
    app.run(debug=True, port=5000)


