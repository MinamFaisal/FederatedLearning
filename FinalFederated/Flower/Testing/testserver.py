import flwr as fl
import tensorflow as tf
import numpy as np
import os
import time

# Define global model
def create_global_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(12,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

global_model = create_global_model()
global_weights_path = ".\Weights\global_weights.h5"
final_model_path = ".\Models\final_global_model.keras"
update_counter_file = "update_counter.txt"

# Load previous weights if available
if os.path.exists(global_weights_path):
    global_model.load_weights(global_weights_path)
    print("✅ Global model weights loaded.")

# Federated learning strategy (but not started immediately)
strategy = fl.server.strategy.FedAvg(
    initial_parameters=fl.common.ndarrays_to_parameters(global_model.get_weights())
)

def monitor_updates():
    """Monitor update counter and start FL when 5 updates are received."""
    while True:
        # Check update count
        if os.path.exists(update_counter_file):
            with open(update_counter_file, "r", encoding="utf-8") as f:  # Force UTF-8
                try:
                    update_count = int(f.read().strip())  # Read and convert to int safely
                except ValueError:
                    update_count = 0  # If error, reset to 0 (corrupt file)

            if update_count >= 5:
                print("🚀 Detected 5 updates. Starting Federated Learning!")

                # Start FL server for ONE round
                fl.server.start_server(
                   server_address="0.0.0.0:8080",
                   strategy=strategy,
                   config=fl.server.ServerConfig(num_rounds=1)
)

    
    

                # Save updated weights
                global_model.save_weights(global_weights_path)
                global_model.save(final_model_path)
                print(f"✅ Global model updated and saved.")

                # Reset counter for next cycle
                with open(update_counter_file, "w", encoding="utf-8") as f:
                    f.write("0")

        time.sleep(2)  # Check every 2 seconds


if __name__ == "__main__":
    print("🕵️ Server is monitoring client updates... Waiting for 5 updates.")
    monitor_updates()
