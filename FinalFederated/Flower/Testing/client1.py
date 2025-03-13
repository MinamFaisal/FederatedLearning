import flwr as fl
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import time
from sklearn.metrics import f1_score, precision_score, recall_score

# -------------------------------------------------------------------
# 1) CLIENT INITIALIZATION
# -------------------------------------------------------------------
client_id = os.getenv("CLIENT_ID", "1")  # Change to "2" for Client 2
model_path = f".\Models\client_model_{client_id}.keras"
update_counter_file = "update_counter.txt"
predict_request_file = f"client{client_id}_predict_request.txt"
results_file = f".\\Results\\results_client{client_id}.csv"

# Load saved model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"✅ [Client {client_id}] Model loaded.")
else:
    print(f"❌ No model found for Client {client_id}. Exiting.")
    exit()

# -------------------------------------------------------------------
# 2) CLIENT CLASS - SENDS/RECEIVES PARAMETERS AND RETAINS KNOWLEDGE
# -------------------------------------------------------------------

def print_model_weights(tag, weights):
    """Print the first few elements of model weights for debugging."""
    print(f"\n🔍 {tag}:")
    for i, w in enumerate(weights):
        print(f"Layer {i}: {w.shape} -> {w.flatten()[:5]} ...")  # Print first 5 values

class MalwareClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        """Send the latest client model parameters to the server."""
        weights = model.get_weights()
        print_model_weights("📤 Client Sending Weights", weights)  # Print before sending
        return [param.tolist() for param in weights]

    def set_parameters(self, parameters):
        """Receive aggregated weights from the server."""
        print_model_weights("📥 Before Updating", model.get_weights())  # Print current weights
        new_weights = [np.array(param) for param in parameters]
        model.set_weights(new_weights)
        model.save(model_path)
        print_model_weights("📥 After Updating", new_weights)  # Print updated weights


    def fit(self, parameters, config):
        """Participate in FL round even if no updates are available."""
        self.set_parameters(parameters)  # ✅ Load latest aggregated weights

        update_count = 0
        if os.path.exists(update_counter_file):
            with open(update_counter_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                update_count = int(content) if content.isdigit() else 0

        if update_count > 0:
            print(f"🔄 [Client {client_id}] Training with new updates.")
            if os.path.exists(results_file):
                df = pd.read_csv(results_file)

                if len(df) > 0:
                    X_train = df.iloc[:, :-1].values.astype(np.float32)
                    X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-7)  # Normalize
                    y_train = (df.iloc[:, -1] == "Malicious").astype(int).values.reshape(-1, 1)

                    # Continue training with previous knowledge
                    model.fit(X_train, y_train, epochs=5, batch_size=4, verbose=1)
                    model.save(model_path)  # Save new trained model
                    print(f"✅ [Client {client_id}] Model trained with updates and saved.")

        else:
            print(f"⚠️ [Client {client_id}] No new updates. Participating in FL round with previous model.")

        return self.get_parameters(config), 1, {}

    def evaluate(self, parameters, config):
        """Handle evaluation after receiving new global model parameters and compute F1 Score."""
        self.set_parameters(parameters)  # ✅ Load latest aggregated weights
        print(f"🔍 [Client {client_id}] Evaluating model...")

        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            if len(df) > 0:
                X_eval = df.iloc[:, :-1].values.astype(np.float32)
                X_eval = (X_eval - X_eval.min(axis=0)) / (X_eval.max(axis=0) - X_eval.min(axis=0) + 1e-7)  # Normalize
                y_true = (df.iloc[:, -1] == "Malicious").astype(int).values  # Convert labels

                # Get predictions
                y_pred = (model.predict(X_eval) > 0.5).astype(np.int32).flatten()

                # Calculate evaluation metrics
                f1 = f1_score(y_true, y_pred, zero_division=1)
                precision = precision_score(y_true, y_pred, zero_division=1)
                recall = recall_score(y_true, y_pred, zero_division=1)

                print(f"🔍 Evaluation - F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                return f1, len(df), {"precision": precision, "recall": recall, "f1_score": f1}

        print(f"⚠️ No evaluation data available.")
        return 0.0, 1, {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

# -------------------------------------------------------------------
# 3) CLIENT LISTENING FOR FL OR USER INPUT
# -------------------------------------------------------------------
def listen():
    while True:
        # 1) Check for prediction requests
        if os.path.exists(predict_request_file):
            print(f"📢 [Client {client_id}] Processing prediction request...")

            features_array = np.load("selected_features.npy")
            
            # ✅ Use the saved model for prediction
            prediction_prob = model.predict(features_array)
            prediction = "Malicious" if prediction_prob[0][0] > 0.5 else "Benign"

            # Save prediction results
            num_features = features_array.shape[1]  # Get the actual feature count dynamically

            df = pd.DataFrame([[*features_array.flatten(), prediction]], 
                  columns=[f"feature_{i}" for i in range(num_features)] + ["prediction"])


            df.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)

            print(f"🔍 Prediction: {prediction}")
            os.remove(predict_request_file)

            # Increment update counter safely
            try:
                with open(update_counter_file, "r+", encoding="utf-8") as f:
                    content = f.read().strip()
                    update_count = int(content) if content.isdigit() else 0  # Ensure valid number
                    update_count += 1  # Increment count
                
                    f.seek(0)  # Move to start of file
                    f.write(str(update_count))  # Write new count
                    f.truncate()  # Ensure no extra content

            except ValueError:
                print(f"⚠️ [Client {client_id}] Error reading update counter. Resetting to 0.")
                with open(update_counter_file, "w", encoding="utf-8") as f:
                    f.write("0")

        # 2) Check if server started FL round
        try:
            with open(update_counter_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                update_count = int(content) if content.isdigit() else 0  # Handle corrupt files

            if update_count >= 5:
                print(f"🛑 [Client {client_id}] Sending updates to server...")

                fl.client.start_client(
                    server_address="127.0.0.1:8080",
                    client=MalwareClient().to_client()
                )
                break  # Stop listening to user input

        except ValueError:
            print(f"⚠️ [Client {client_id}] Corrupt counter file detected. Resetting to 0.")
            with open(update_counter_file, "w", encoding="utf-8") as f:
                f.write("0")

        time.sleep(3)

# -------------------------------------------------------------------
# 4) START CLIENT
# -------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 [Client {client_id}] Listening for user input...")
    listen()
