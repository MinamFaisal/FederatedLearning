import os
import fitz  # PyMuPDF
import pandas as pd

def is_pdf(file_path):
    """
    Check if the file is a valid PDF by reading its signature.
    PDF files typically start with '%PDF'.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
        return header == b'%PDF'
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return False

def extract_features(file_path):
    """
    Extract static features from a PDF file.
    Features include:
    - File size
    - Number of pages
    - Presence of JavaScript
    - Number of images
    - Suspicious URLs
    - Number of embedded files
    - Presence of Launch Actions
    - Presence of Open Actions
    - Number of objects
    - Presence of XFA Forms
    - Suspicious metadata
    - Presence of compressed streams
    """
    features = {
        "file_name": os.path.basename(file_path),
        "file_size": os.path.getsize(file_path),  # File size in bytes
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
            features["contains_javascript"] = doc.is_encrypted  # Check for encryption (could indicate malicious intent)
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

def process_files(folder_path, label):
    """
    Process all files in a folder, extract features, and add a label.
    """
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
                    "errors": "Not a valid PDF file or invalid structure."
                }
            features["label"] = label
            all_features.append(features)
    return all_features

if __name__ == "__main__":
    # Define paths
    benign_folder = r"F:\7th Semester\FYP\Dataset\training_cleanpdfs"  # Path to Benign folder
    malicious_folder = r"F:\7th Semester\FYP\Dataset\training_malpdfs"  # Path to Malicious folder
    output_csv ="trainingfeatures.csv"

    # Process benign and malicious files
    print("Processing benign files...")
    benign_features = process_files(benign_folder, label="benign")

    print("\nProcessing malicious files...")
    malicious_features = process_files(malicious_folder, label="malicious")

    # Combine features
    all_features = benign_features + malicious_features

    # Save to CSV
    selected_features = [
        "file_name", "file_size", "num_pages", "contains_javascript", 
        "num_images", "suspicious_urls", "num_embedded_files", "contains_launch_action",
        "contains_open_action", "num_objects", "contains_xfa_forms", 
        "contains_suspicious_metadata", "contains_compressed_streams", "label"
    ]
    df = pd.DataFrame(all_features)
    df = df[selected_features]  # Ensure all selected columns are included
    df.to_csv(output_csv, index=False)
    print(f"\nFeatures saved to '{output_csv}'.")
