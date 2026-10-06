import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def train_traffic_pipeline(data_path="historical_traffic_data.csv", model_output="traffic_model.pkl", metrics_output="model_metrics.txt"):
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find '{data_path}'. Please run 'python generate_data.py' first to build it!")
        return

    print("📊 Step 1: Loading historical traffic logs into training memory dataframe...")
    df = pd.read_csv(data_path)

    print("⚙️ Step 2: Extracting dataset attributes and processing labels...")
    # Initialize separate Label Encoders to change categorical strings into numerical data features safely
    weather_encoder = LabelEncoder()
    target_encoder = LabelEncoder()

    df['Weather_Encoded'] = weather_encoder.fit_transform(df['Weather_Condition'])
    df['Target_Encoded'] = target_encoder.fit_transform(df['Congestion_Level'])

    # Define the exact features used for mathematical training split computations
    feature_columns = ['Hour', 'Is_Weekend', 'Is_Holiday', 'Junction_Activity_Weight', 'Weather_Encoded', 'Vehicle_Count']
    X = df[feature_columns]
    y = df['Target_Encoded']

    # Partition dataset into 80% Training sets and 20% Evaluation Test matrix states
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🔮 Step 3: Initializing and fitting the Random Forest Classifier ensemble logic...")
    # Using 100 estimator trees with a fixed seed to establish optimal classification bounds
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    print("⏱️ Step 4: Assessing structural validation accuracy and cross-testing weights...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)

    print(f"✅ Training completed! Out-of-Sample Backtesting Accuracy: {accuracy * 100:.2f}%")

    print("📦 Step 5: Exporting artifacts and packing serializations...")
    # Wrap model layers, label maps, and features list together inside a portable dictionary framework object
    model_payload = {
        "model_engine": model,
        "weather_mapper": weather_encoder,
        "target_mapper": target_encoder,
        "features_list": feature_columns
    }

    # Save the pipeline as a serial binary file inside your folder workspace paths
    with open(model_output, 'wb') as f:
        pickle.dump(model_payload, f)

    # Export a text file summarizing the project's verification report metrics
    with open(metrics_output, 'w') as f:
        f.write(f"=== Project Model Verification Report ===\n")
        f.write(f"Generated Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Global Model Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(f"=== Precision and Recall Breakdown Matrix ===\n")
        f.write(report)
        f.write("\n=== Relative Feature Importance Array ===\n")
        for col, score in zip(feature_columns, model.feature_importances_):
            f.write(f"{col}: {score:.4f}\n")

    print(f"💾 Portable model pipeline packed successfully: {model_output}")
    print(f"📝 Backtesting documentation report written: {metrics_output}")

if __name__ == "__main__":
    train_traffic_pipeline()
