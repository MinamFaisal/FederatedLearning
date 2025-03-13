import flwr as fl
import tensorflow as tf
import numpy as np
import os

# Define global model
def create_global_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
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
global_weights_path = ".\\Trained_weights\\global_weights.h5"  # ✅ Save weights in .h5 format
final_model_path = ".\\Trained_models\\final_global_model.keras"

# Define standard Federated Averaging strategy
strategy = fl.server.strategy.FedAvg(
    initial_parameters=fl.common.ndarrays_to_parameters(global_model.get_weights())
)


# Start the server
if __name__ == "__main__":
    num_rounds = 3

    print("🚀 Starting Federated Learning Server...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )

    # ✅ Use TensorFlow's save_weights() instead of np.save()
    global_model.save_weights(global_weights_path)  # Save weights in .h5 format
    global_model.save(final_model_path)  # Save full model
    print(f"✅ Final global model saved at {final_model_path} and weights at {global_weights_path}")
