import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score

# Define test folder paths
TEST_MALICIOUS_FOLDER = "../../../testing_malpdfs"
TEST_BENIGN_FOLDER = "../../../testing_cleanpdfs"

# Load scalers and models
scaler_client1 = joblib.load("./Trained_scalars/scaler_client_1.pkl")
scaler_client2 = joblib.load("./Trained_scalars/scaler_client_2.pkl")

model_client1 = tf.keras.models.load_model("./Trained_models/client_model_1.keras")
model_client2 = tf.keras.models.load_model("./Trained_models/client_model_2.keras")


def is_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
        return header == b'%PDF'
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return False

def calculate_entropy(file_path):
    """Calculate the entropy of a file."""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        byte_counts = collections.Counter(data)
        total_bytes = len(data)
        entropy = -sum((count / total_bytes) * math.log2(count / total_bytes) for count in byte_counts.values())
        return entropy
    except Exception:
        return -1  # Return -1 if entropy calculation fails

def extract_features(file_path):
    features = {
        "file_name": os.path.basename(file_path),
        "file_size": os.path.getsize(file_path),
        "num_pages": 0,
        "contains_javascript": False,
        "num_images": 0,
        "suspicious_urls": 0,
        "num_embedded_files": 0,
        "contains_launch_action": False,
        "contains_open_action": False,
        "num_objects": 0,
        "contains_xfa_forms": False,
        "contains_suspicious_metadata": False,
        "contains_compressed_streams": False,
        "entropy": calculate_entropy(file_path),
        "num_annotations": 0,
        "num_embedded_fonts": 0,
        "contains_acroform": False,
        "num_suspicious_keywords": 0,
        "num_encrypted_objects": 0,
        "num_trailers": 0,
        "contains_obfuscated_js": False,
        "errors": None,
    }
    
    try:
        with fitz.open(file_path) as doc:
            features["num_pages"] = len(doc)
            features["contains_javascript"] = doc.is_encrypted
            features["num_images"] = sum(1 for page in doc for img in page.get_images())
            features["num_annotations"] = sum(1 for page in doc for _ in page.annots() or [])
            features["num_embedded_files"] = len(doc.embfile_names())
            features["num_objects"] = len(doc.xref_objects())
            features["num_trailers"] = len([obj for obj in doc.xref_objects().values() if obj.get("/Root")])
            
            suspicious_keywords = ["/JS", "/JavaScript", "/AA", "/OpenAction", "/Launch"]
            
            suspicious_count = 0
            for obj_id, obj in doc.xref_objects().items():
                obj_str = str(obj)
                if any(keyword in obj_str for keyword in suspicious_keywords):
                    suspicious_count += 1
                if "XFA" in obj_str:
                    features["contains_xfa_forms"] = True
                if "/Font" in obj_str:
                    features["num_embedded_fonts"] += 1
                if "/Encrypt" in obj_str:
                    features["num_encrypted_objects"] += 1
                if "eval(" in obj_str or "unescape(" in obj_str:
                    features["contains_obfuscated_js"] = True
            features["num_suspicious_keywords"] = suspicious_count
            
            metadata = doc.metadata
            if metadata:
                for key, value in metadata.items():
                    if any(suspicious in value.lower() for suspicious in ["malware", "hacked", "exploit"]):
                        features["contains_suspicious_metadata"] = True
                        break
    
    except Exception as e:
        features["errors"] = str(e)
    
    return features


# Extract features from test PDFs
def extract_features_from_folder(folder, label):
    data = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path) and is_pdf(file_path):
            features = extract_features(file_path)
            features["label"] = label
            data.append(features)
    return pd.DataFrame(data)

# Get test data
malicious_df = extract_features_from_folder(TEST_MALICIOUS_FOLDER, 1)  # Label 1 for malicious
benign_df = extract_features_from_folder(TEST_BENIGN_FOLDER, 0)  # Label 0 for benign
test_data = pd.concat([malicious_df, benign_df], ignore_index=True)

# Prepare feature matrix
boolean_features = [
    "contains_javascript", "contains_launch_action", "contains_open_action",
    "contains_xfa_forms", "contains_suspicious_metadata", "contains_compressed_streams",
    "contains_acroform", "contains_obfuscated_js"
]

test_data[boolean_features] = test_data[boolean_features].astype(int)

features = test_data[
    ["file_size", "num_pages", "contains_javascript", "num_images", "suspicious_urls",
    "num_embedded_files", "contains_launch_action", "contains_open_action",
    "num_objects", "contains_xfa_forms", "contains_suspicious_metadata",
    "contains_compressed_streams", "entropy", "num_annotations",
    "num_embedded_fonts", "contains_acroform", "num_suspicious_keywords",
    "num_encrypted_objects", "num_trailers", "contains_obfuscated_js"]
].values
labels = test_data["label"].values

# Scale features for both models
features_scaled_client1 = scaler_client1.transform(features)
features_scaled_client2 = scaler_client2.transform(features)

# Make predictions
predictions_client1 = (model_client1.predict(features_scaled_client1) > 0.5).astype(int)
predictions_client2 = (model_client2.predict(features_scaled_client2) > 0.5).astype(int)

# Compute accuracies
accuracy_client1 = accuracy_score(labels, predictions_client1)
accuracy_client2 = accuracy_score(labels, predictions_client2)

print(f"✅ Client 1 Model Accuracy: {accuracy_client1:.4f}")
print(f"✅ Client 2 Model Accuracy: {accuracy_client2:.4f}")
