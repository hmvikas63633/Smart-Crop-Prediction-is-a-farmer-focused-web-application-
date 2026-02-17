"""
Train the crop prediction model using the Kaggle dataset
This script should be run once to train and save the model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(csv_path='Crop_recommendation.csv'):
    """
    Train the Random Forest model for crop prediction
    
    Args:
        csv_path: Path to the crop recommendation CSV file
    """
    
    print("=" * 60)
    print("CROP PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load the dataset
    print(f"\n1. Loading dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"   ⚠️  Dataset not found at {csv_path}")
        print(f"   📥 Please download 'Crop_recommendation.csv' from Kaggle")
        print(f"   🔗 https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        return
    
    df = pd.read_csv(csv_path)
    print(f"   ✅ Dataset loaded successfully!")
    print(f"   📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display dataset info
    print(f"\n2. Dataset Information:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Unique crops: {df['label'].nunique()}")
    print(f"   Crops: {sorted(df['label'].unique())}")
    
    # Prepare features and target
    print(f"\n3. Preparing data for training...")
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Train the model
    print(f"\n4. Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print(f"   ✅ Model trained successfully!")
    
    # Evaluate the model
    print(f"\n5. Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   🎯 Accuracy: {accuracy * 100:.2f}%")
    
    # Feature importance
    print(f"\n6. Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']:15s}: {row['importance']:.4f}")
    
    # Save the model
    model_path = 'crop_model.pkl'
    print(f"\n7. Saving model to: {model_path}")
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved successfully!")
    
    # Test prediction
    print(f"\n8. Testing prediction with sample data...")
    sample_input = [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]
    prediction = model.predict(sample_input)
    probabilities = model.predict_proba(sample_input)[0]
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    
    print(f"   Sample input: N=90, P=42, K=43, Temp=20.8, Humidity=82, pH=6.5, Rainfall=202.9")
    print(f"   Predicted crop: {prediction[0]}")
    print(f"   Top 3 predictions:")
    for idx in top_3_idx:
        print(f"      {model.classes_[idx]:15s}: {probabilities[idx]*100:.2f}%")
    
    print(f"\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📝 Next steps:")
    print(f"   1. Run: python app.py")
    print(f"   2. Open browser: http://localhost:5000")
    print(f"   3. Start making predictions!")
    print()

if __name__ == '__main__':
    # Check if CSV file exists in current directory
    csv_file = 'Crop_recommendation.csv'
    
    if os.path.exists(csv_file):
        train_model(csv_file)
    else:
        print("\n" + "=" * 60)
        print("⚠️  DATASET NOT FOUND")
        print("=" * 60)
        print(f"\nPlease download the dataset from Kaggle:")
        print(f"🔗 https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print(f"\nDownload 'Crop_recommendation.csv' and place it in this directory:")
        print(f"📁 {os.getcwd()}")
        print(f"\nThen run this script again: python train_model.py")
        print()
