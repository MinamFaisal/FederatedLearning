import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# 1) LOAD AND BALANCE DATASET
# -------------------------------------------------------------------
client_id = os.getenv("CLIENT_ID", "1")
file_path = f'.\\Pre_Processing\\client_{client_id}.csv'
data = pd.read_csv(file_path)

# Balance dataset
malicious_df = data[data["label"] == "malicious"]
benign_df = data[data["label"] == "benign"]
min_count = min(len(malicious_df), len(benign_df))

# Undersample to balance
malicious_df_bal = malicious_df.sample(n=min_count, random_state=42)
benign_df_bal = benign_df.sample(n=min_count, random_state=42)
data_balanced = pd.concat([malicious_df_bal, benign_df_bal], ignore_index=True)

print(f"🔎 [Client {client_id}] Balanced dataset: {len(data_balanced)} samples")

# -------------------------------------------------------------------
# 2) FEATURE PROCESSING
# -------------------------------------------------------------------
boolean_features = [
    "contains_javascript", "contains_launch_action", "contains_open_action",
    "contains_xfa_forms", "contains_suspicious_metadata", "contains_compressed_streams",
    "contains_acroform", "contains_obfuscated_js"
]

data_balanced[boolean_features] = data_balanced[boolean_features].astype(int)

features = data_balanced[[  # Select features
    "file_size", "num_pages", "contains_javascript", "num_images", "suspicious_urls",
    "num_embedded_files", "contains_launch_action", "contains_open_action",
    "num_objects", "contains_xfa_forms", "contains_suspicious_metadata",
    "contains_compressed_streams", "entropy", "num_annotations",
    "num_embedded_fonts", "contains_acroform", "num_suspicious_keywords",
    "num_encrypted_objects", "num_trailers", "contains_obfuscated_js"
]].values

labels = data_balanced["label"].apply(lambda x: 1 if x == "malicious" else 0).values



# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
joblib.dump(scaler, f".\\Trained_scalars\\scaler_client_{client_id}.pkl")

# -------------------------------------------------------------------
# 3) MODEL INITIALIZATION
# -------------------------------------------------------------------
model_path = f".\\Trained_models\\client_model_{client_id}.keras"
weights_path = f".\\Trained_weights\\client_weights_{client_id}.h5"

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Load or create model
if os.path.exists(model_path):
    print(f"🔄 Loading existing model for Client {client_id}...")
    model = tf.keras.models.load_model(model_path)
else:
    print(f"🆕 Creating a new model for Client {client_id}...")
    model = create_model()

# -------------------------------------------------------------------
# 4) INITIAL LOCAL TRAINING BEFORE FEDERATED LEARNING
# -------------------------------------------------------------------
local_epochs = 10  # Number of local training epochs before FL starts

print(f"🏋️‍♂️ [Client {client_id}] Performing initial local training before FL begins...")

model.fit(features_scaled, labels, epochs=local_epochs, batch_size=32, verbose=1)

# ✅ Save the locally trained model before FL starts
model.save(model_path)
model.save_weights(weights_path)

print(f"✅ [Client {client_id}] Initial local training complete. Model saved at {model_path} and weights at {weights_path}")

# -------------------------------------------------------------------
# 5) DEFINE CLIENT FOR FEDERATED LEARNING
# -------------------------------------------------------------------
class MalwareClient(fl.client.NumPyClient):

    def __init__(self):
        super().__init__()
        self.model = model
        self.current_round = 0  # Track federated rounds
        self.total_rounds = int(os.getenv("TOTAL_ROUNDS", 3))  # Total rounds

    def get_parameters(self, config):
        """Send model weights to the server."""
        self.current_round += 1  # Increase round count
        
        # ✅ Save model and weights **only** at the last round
        if self.current_round == self.total_rounds:
            self.model.save(model_path)
            self.model.save_weights(weights_path)
            print(f"✅ Final Model saved at {model_path} and weights at {weights_path}")

        return [param.tolist() for param in self.model.get_weights()]

    def set_parameters(self, parameters):
        """Receive global model weights from server."""
        self.model.set_weights([np.array(param) for param in parameters])

    def fit(self, parameters, config):
        """Train model after receiving global weights."""
        self.set_parameters(parameters)  # Receive updated global weights

        print(f"🔄 Training model for Client {client_id} after receiving new global weights...")

        # ✅ Perform local training for a few epochs before sending updates
        self.model.fit(features_scaled, labels, epochs=5, batch_size=32, verbose=1)

        return self.get_parameters(config), len(features_scaled), {}

    def evaluate(self, parameters, config):
        """Evaluate the model."""
        self.set_parameters(parameters)  # Receive final global weights

        loss, accuracy = self.model.evaluate(features_scaled, labels, verbose=0)
        print(f"🔍 Final evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return loss, len(features_scaled), {"accuracy": accuracy}

# -------------------------------------------------------------------
# 6) START CLIENT
# -------------------------------------------------------------------
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=MalwareClient()
)
