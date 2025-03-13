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
def extract_features(file_path):
    """Extract features from a PDF file for analysis."""
    features = {
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
        "errors": None,
    }
    
    try:
        with fitz.open(file_path) as doc:
            features["num_pages"] = len(doc)
            features["contains_javascript"] = doc.is_encrypted  # Encryption check (potentially suspicious)
            features["num_images"] = sum(1 for page in doc for img in page.get_images())

            # Identify suspicious URLs
            suspicious_urls = []
            for page in doc:
                text = page.get_text("text")
                suspicious_urls.extend([url for url in text.split() if "http" in url])
            features["suspicious_urls"] = len(suspicious_urls)

            # Count number of embedded files
            features["num_embedded_files"] = len(doc.embfile_names())

            # Check for Launch and Open actions
            for obj in doc:
                annots = obj.annots()
                if annots:
                    for annot in annots:
                        if "/Launch" in annot.info:
                            features["contains_launch_action"] = True
                        if "/OpenAction" in annot.info:
                            features["contains_open_action"] = True

            # Count number of objects
            features["num_objects"] = len(doc.xref_objects())

            # Check for XFA Forms
            try:
                for obj_id, obj in doc.xref_objects().items():
                    if isinstance(obj, tuple) and len(obj) > 1 and isinstance(obj[1], str) and "XFA" in obj[1]:
                        features["contains_xfa_forms"] = True
                        break
            except Exception as e:
                features["errors"] = f"XFA check failed: {e}"

            # Check for suspicious metadata
            metadata = doc.metadata
            if metadata:
                for key, value in metadata.items():
                    if any(suspicious in value.lower() for suspicious in ["malware", "hacked", "exploit"]):
                        features["contains_suspicious_metadata"] = True
                        break

            # Check for compressed streams
            for obj in doc:
                for stream in obj.get_stm():
                    if stream.is_compressed:
                        features["contains_compressed_streams"] = True
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
    "contains_xfa_forms", "contains_suspicious_metadata", "contains_compressed_streams"
]
for feature in boolean_features:
    extracted_features[feature] = int(extracted_features[feature])  # Convert True/False to 1/0

# Convert dictionary to NumPy array for prediction
features_array = np.array([[  
    extracted_features["file_size"], extracted_features["num_pages"], extracted_features["contains_javascript"],
    extracted_features["num_images"], extracted_features["suspicious_urls"], extracted_features["num_embedded_files"],
    extracted_features["contains_launch_action"], extracted_features["contains_open_action"],
    extracted_features["num_objects"], extracted_features["contains_xfa_forms"],
    extracted_features["contains_suspicious_metadata"], extracted_features["contains_compressed_streams"]
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