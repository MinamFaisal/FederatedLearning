import numpy as np
import os
import random
import pandas as pd
import time
import fitz  # PyMuPDF for PDF processing

# -------------------------------------------------------------------
# 1) USER UPLOADS A PDF FILE FOR ANALYSIS
# -------------------------------------------------------------------
pdf_file = input("📄 Enter the path of the PDF file for analysis: ")

if not os.path.exists(pdf_file):
    print("❌ Error: File does not exist.")
    exit()

# -------------------------------------------------------------------
# 2) FEATURE EXTRACTION FUNCTION
# -------------------------------------------------------------------
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
# -------------------------------------------------------------------
# 3) EXTRACT FEATURES FROM PDF FILE
# -------------------------------------------------------------------
extracted_features = extract_features(pdf_file)

# Convert boolean features to integers for model compatibility
boolean_features = [
    "contains_javascript", "contains_launch_action", "contains_open_action",
    "contains_xfa_forms", "contains_suspicious_metadata", "contains_compressed_streams",
    "contains_acroform", "contains_obfuscated_js"
]
for feature in boolean_features:
    extracted_features[feature] = int(extracted_features[feature])  # Convert True/False to 1/0

# Convert dictionary to NumPy array for prediction
# Convert extracted features into a NumPy array for prediction
features_array = np.array([[  
    extracted_features["file_size"], extracted_features["num_pages"], extracted_features["contains_javascript"],
    extracted_features["num_images"], extracted_features["suspicious_urls"], extracted_features["num_embedded_files"],
    extracted_features["contains_launch_action"], extracted_features["contains_open_action"],
    extracted_features["num_objects"], extracted_features["contains_xfa_forms"],
    extracted_features["contains_suspicious_metadata"], extracted_features["contains_compressed_streams"],
    extracted_features["entropy"], extracted_features["num_annotations"], extracted_features["num_embedded_fonts"],
    extracted_features["contains_acroform"], extracted_features["num_suspicious_keywords"],
    extracted_features["num_encrypted_objects"], extracted_features["num_trailers"],
    extracted_features["contains_obfuscated_js"]
]])


print("✅ Extracted PDF features.")

# -------------------------------------------------------------------
# 4) SELECT A RANDOM CLIENT TO HANDLE PREDICTION
# -------------------------------------------------------------------
selected_client = random.choice(["client1", "client2"])
predict_request_file = f"{selected_client}_predict_request.txt"

# Save features for the selected client
np.save("selected_features.npy", features_array)

# Select a random client
selected_client = random.choice(["client1", "client2"])
predict_request_file = f"{selected_client}_predict_request.txt"

open(predict_request_file, "w").close()
print(f"📢 Request sent to {selected_client}, waiting for prediction...")

# Wait for client response
results_file = f".\\Results\\results_{selected_client}.csv"
while not os.path.exists(results_file):
    time.sleep(2)

# Read last prediction
df = pd.read_csv(results_file)
print(f"🔍 Final Prediction: {df.iloc[-1]['prediction']}")
