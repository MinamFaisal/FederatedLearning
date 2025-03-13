import os
import shutil
import numpy as np
import tensorflow as tf
from PyPDF2 import PdfReader
import filetype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define dataset paths
CLEAN_PDFS_DIR = r"F:\7th Semester\FYP\Dataset\CLEAN_PDF_9000_files"
MALICIOUS_PDFS_DIR = r"F:\7th Semester\FYP\Dataset\MALWARE_PDF_PRE_04-2011_10982_files"

CLEAN_PDF_FOLDER = r"F:\7th Semester\FYP\Dataset\clean_pdf"
MAL_PDF_FOLDER = r"F:\7th Semester\FYP\Dataset\mal_pdf"

TRAINING_CLEAN_DIR = r"F:\7th Semester\FYP\Dataset\training_cleanpdfs"
TRAINING_MAL_DIR = r"F:\7th Semester\FYP\Dataset\training_malpdfs"
TESTING_CLEAN_DIR = r"F:\7th Semester\FYP\Dataset\testing_cleanpdfs"
TESTING_MAL_DIR = r"F:\7th Semester\FYP\Dataset\testing_malpdfs"
CORRUPTED_PDFS_DIR = r"F:\7th Semester\FYP\Dataset\corrupted_pdfs"  # New directory for corrupted PDFs

# Create necessary directories
def create_directories():
    required_dirs = [
        CLEAN_PDF_FOLDER, MAL_PDF_FOLDER,
        TRAINING_CLEAN_DIR, TRAINING_MAL_DIR, TESTING_CLEAN_DIR, TESTING_MAL_DIR,
        CORRUPTED_PDFS_DIR,  # Ensure the corrupted folder exists
    ]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created missing directory: {dir_path}")

# Function to move corrupted files to the corrupted folder
def move_to_corrupted(pdf_path):
    try:
        corrupted_folder = CORRUPTED_PDFS_DIR
        shutil.move(pdf_path, os.path.join(corrupted_folder, os.path.basename(pdf_path)))
        print(f"Moved corrupted file: {pdf_path}")
    except Exception as e:
        print(f"Error moving corrupted file {pdf_path}: {e}")

# Update the is_corrupted_pdf function to ignore startxref error
def is_corrupted_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            # Try to read the PDF and check for common errors
            pdf_reader.pages
    except Exception as e:
        # Ignore the startxref error and continue processing
        if 'startxref' in str(e):
            print(f"Ignoring startxref error for {pdf_path}")
            return False  # Don't consider this file as corrupted, just skip this error
        # Handle other errors
        if 'EOF' in str(e):
            print(f"Corrupted file detected: {pdf_path} due to EOF error")
            return True  # Treat EOF errors as corrupted
        if 'Invalid parent xref' in str(e):
            print(f"Corrupted file detected: {pdf_path} Invalid parent xref error")
            return True  # Treat Invalid parent xref errors as corrupted
        if 'incorrect startxref pointer' in str(e):
            print(f"Corrupted file detected: {pdf_path} due to incorrect startxref pointer error")
            return True  # Treat incorrect startxref pointer errors as corrupted
        

        print(f"Error reading {pdf_path}: {e}")
        return True
        
    return False



# Function to prepare dataset
def prepare_and_move_data():
    # Process clean PDFs
    clean_files = [f for f in os.listdir(CLEAN_PDFS_DIR) if f.endswith('.pdf')]
    for file in clean_files:
        pdf_path = os.path.join(CLEAN_PDFS_DIR, file)
        if is_corrupted_pdf(pdf_path):
            move_to_corrupted(pdf_path)
        else:
            shutil.move(pdf_path, os.path.join(CLEAN_PDF_FOLDER, file))

    # Process malicious PDFs
    mal_files = [f for f in os.listdir(MALICIOUS_PDFS_DIR)]
    for file in mal_files:
        pdf_path = os.path.join(MALICIOUS_PDFS_DIR, file)
        if is_corrupted_pdf(pdf_path):
            move_to_corrupted(pdf_path)
        else:
            shutil.move(pdf_path, os.path.join(MAL_PDF_FOLDER, file))

# Split dataset into training and testing sets
def split_and_move_data():
    # Create necessary directories
    create_directories()
    
    # Get list of files in the directories
    clean_files = [f for f in os.listdir(CLEAN_PDF_FOLDER) if f.endswith('.pdf')]
    mal_files = [f for f in os.listdir(MAL_PDF_FOLDER)]

    # Split dataset into training and testing
    clean_train, clean_test = train_test_split(clean_files, test_size=0.2, random_state=42)
    mal_train, mal_test = train_test_split(mal_files, test_size=0.2, random_state=42)

    # Move files to training folders
    for file in clean_train:
        shutil.move(os.path.join(CLEAN_PDF_FOLDER, file), TRAINING_CLEAN_DIR)
        print(f"Moved {file} to training_cleanpdfs")

    for file in mal_train:
        shutil.move(os.path.join(MAL_PDF_FOLDER, file), TRAINING_MAL_DIR)
        print(f"Moved {file} to training_malpdfs")

    # Move files to testing folders
    for file in clean_test:
        shutil.move(os.path.join(CLEAN_PDF_FOLDER, file), TESTING_CLEAN_DIR)
        print(f"Moved {file} to testing_cleanpdfs")

    for file in mal_test:
        shutil.move(os.path.join(MAL_PDF_FOLDER, file), TESTING_MAL_DIR)
        print(f"Moved {file} to testing_malpdfs")

    print("Data moved and split successfully.")


# Start the process
create_directories()
prepare_and_move_data()
split_and_move_data()
#federated_training()
