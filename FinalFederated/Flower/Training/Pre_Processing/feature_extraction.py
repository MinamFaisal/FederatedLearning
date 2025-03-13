import os
import fitz  # PyMuPDF
import pandas as pd
import math
import collections

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

def process_files(folder_path, label):
    all_features = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_pdf(file_path):
                print(f"Processing valid PDF: {file_path}")
                features = extract_features(file_path)
            else:
                print(f"Skipping invalid or non-PDF file: {file_path}")
                features = {
                    "file_name": os.path.basename(file_path),
                    "file_size": 0,
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
                    "entropy": -1,
                    "num_annotations": 0,
                    "num_embedded_fonts": 0,
                    "contains_acroform": False,
                    "num_suspicious_keywords": 0,
                    "num_encrypted_objects": 0,
                    "num_trailers": 0,
                    "contains_obfuscated_js": False,
                    "errors": "Not a valid PDF file or invalid structure."
                }
            features["label"] = label
            all_features.append(features)
    return all_features

if __name__ == "__main__":
    benign_folder = "../../../../training_cleanpdfs"
    malicious_folder = "../../../../training_malpdfs"
    output_csv = "trainingfeatures.csv"

    print("Processing benign files...")
    benign_features = process_files(benign_folder, label="benign")
    print("\nProcessing malicious files...")
    malicious_features = process_files(malicious_folder, label="malicious")

    all_features = benign_features + malicious_features

    selected_features = [
        "file_name", "file_size", "num_pages", "contains_javascript", "num_images", 
        "suspicious_urls", "num_embedded_files", "contains_launch_action", "contains_open_action", 
        "num_objects", "contains_xfa_forms", "contains_suspicious_metadata", "contains_compressed_streams", 
        "entropy", "num_annotations", "num_embedded_fonts", "contains_acroform", 
        "num_suspicious_keywords", "num_encrypted_objects", "num_trailers", "contains_obfuscated_js", "label"
    ]
    df = pd.DataFrame(all_features)
    df = df[selected_features]
    df.to_csv(output_csv, index=False)
    print(f"\nFeatures saved to '{output_csv}'.")
