import flwr as fl
import tensorflow as tf
import numpy as np
import os
import time

# -------------------------------------------------------------------
# 1) DEFINE GLOBAL MODEL
# -------------------------------------------------------------------
def create_global_model():
    """Define a simple neural network for federated learning."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

global_model = create_global_model()
global_weights_path = "./Weights/global_weights.h5"
final_model_path = "./Models/final_global_model.keras"
update_counter_file = "update_counter.txt"

# Load previous global weights if available
if os.path.exists(global_weights_path):
    global_model.load_weights(global_weights_path)
    print("✅ Global model weights loaded.")

# -------------------------------------------------------------------
# 2) DEBUG FUNCTION - PRINT MODEL WEIGHTS
# -------------------------------------------------------------------
def print_model_weights(stage, weights):
    """Print model weights at different FL stages."""
    print(f"\n🔍 {stage} Weights (Server):")
    for i, w in enumerate(weights):
        print(f"Layer {i}: {w.shape} -> {w.flatten()[:5]} ...")  # Print first 5 values per layer

# -------------------------------------------------------------------
# 3) DEFINE FEDERATED STRATEGY WITH CORRECT AGGREGATION
# -------------------------------------------------------------------
class CustomFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy to manually apply weight updates."""
    
    def aggregate_fit(
        self, rnd, results, failures
    ):
        """Aggregate model updates from clients and ensure correct averaging."""
        if failures:
            print(f"⚠️ {len(failures)} client(s) failed.")

        # Convert parameters to numpy arrays
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)

        # Convert back to ndarrays
        aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters[0])
        
        # Debug: Print the aggregated weights after averaging
        print_model_weights("📊 After Aggregation", aggregated_weights)

        # Update global model with new weights
        global_model.set_weights(aggregated_weights)
        global_model.save_weights(global_weights_path)
        global_model.save(final_model_path)
        print(f"✅ Global model updated and saved.")
        
        return aggregated_parameters

# -------------------------------------------------------------------
# 4) MONITOR CLIENT UPDATES AND START FEDERATED LEARNING
# -------------------------------------------------------------------
def monitor_updates():
    """Monitor update counter and start FL when 5 updates are received."""
    while True:
        if os.path.exists(update_counter_file):
            with open(update_counter_file, "r", encoding="utf-8") as f:
                try:
                    update_count = int(f.read().strip())
                except ValueError:
                    update_count = 0

            if update_count >= 5:
                print("🚀 Detected 5 updates. Starting Federated Learning!")

                # Print global model weights before aggregation
                print_model_weights("📩 Received Before Aggregation", global_model.get_weights())

                # Start FL server with custom FedAvg
                fl.server.start_server(
                    server_address="0.0.0.0:8080",
                    strategy=CustomFedAvg(initial_parameters=fl.common.ndarrays_to_parameters(global_model.get_weights())),
                    config=fl.server.ServerConfig(num_rounds=1)
                )

                # Reset update counter after training
                with open(update_counter_file, "w", encoding="utf-8") as f:
                    f.write("0")

        time.sleep(2)

# -------------------------------------------------------------------
# 5) START SERVER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("🕵️ Server is monitoring client updates... Waiting for 5 updates.")
    monitor_updates()
