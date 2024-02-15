import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Quantum computing imports with fallback
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram
    from qiskit.execute_function import execute  # Updated import for execute
    QUANTUM_AVAILABLE = True
except ImportError:
    print("Qiskit or Qiskit-aer not fully installed. Using mock quantum simulation.")
    QUANTUM_AVAILABLE = False

class QuantumInspiredNeuralNetwork:
    def __init__(self, input_dim, quantum_layers=3, classical_layers=2):
        """
        A hybrid quantum-classical neural network simulator
        
        Parameters:
        - input_dim: Dimension of input data
        - quantum_layers: Number of quantum-inspired layers
        - classical_layers: Number of classical neural network layers
        """
        self.input_dim = input_dim
        self.quantum_layers = quantum_layers
        self.classical_layers = classical_layers
        
        # Quantum backend simulation
        if QUANTUM_AVAILABLE:
            self.quantum_backend = AerSimulator(method='statevector')
        else:
            self.quantum_backend = None
        
        # Classical neural network model
        self.classical_model = self._build_classical_model()
        
    def _create_quantum_layer(self, num_qubits):
        """
        Create a quantum-inspired layer using mock or Qiskit simulation
        
        Parameters:
        - num_qubits: Number of qubits in the quantum circuit
        
        Returns:
        - Quantum circuit representation of the layer
        """
        if QUANTUM_AVAILABLE:
            qc = QuantumCircuit(num_qubits)
            
            # Apply random rotations and entanglement
            for qubit in range(num_qubits):
                # Random rotation gates
                qc.rx(np.random.uniform(0, np.pi), qubit)
                qc.ry(np.random.uniform(0, np.pi), qubit)
            
            # Entanglement using CX gates (CNOT)
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            
            return qc
        else:
            # Mock quantum layer for demonstration
            return None
    
    def _build_classical_model(self):
        """
        Build a classical neural network model
        
        Returns:
        - Keras sequential model
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.InputLayer(input_shape=(self.input_dim,)))
        
        # Hidden layers
        for _ in range(self.classical_layers):
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.2))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(optimizer='adam', 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    
    def quantum_feature_map(self, classical_input):
        """
        Convert classical input to quantum feature representation
        
        Parameters:
        - classical_input: Input data from classical domain
        
        Returns:
        - Quantum feature map representation
        """
        if not QUANTUM_AVAILABLE:
            # Fallback to classical feature mapping
            return classical_input
        
        # Determine number of qubits based on input dimension
        num_qubits = int(np.ceil(np.log2(self.input_dim)))
        
        # Create quantum circuit
        qc = QuantumCircuit(num_qubits)
        
        # Encode classical input into quantum states
        for i, value in enumerate(classical_input):
            if i < num_qubits:
                # Rotate qubit based on input value
                qc.ry(value * np.pi, i)
        
        # Add quantum layer transformations
        for _ in range(self.quantum_layers):
            layer = self._create_quantum_layer(num_qubits)
            if layer:
                qc.compose(layer, inplace=True)
        
        return qc
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the hybrid quantum-classical neural network
        
        Parameters:
        - X_train: Training input data
        - y_train: Training labels
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        """
        # Quantum feature mapping for training data
        quantum_features = []
        
        if QUANTUM_AVAILABLE:
            for sample in X_train:
                qc = self.quantum_feature_map(sample)
                # Simulate quantum circuit
                job = execute(qc, self.quantum_backend)
                result = job.result()
                state_vector = result.get_statevector()
                quantum_features.append(np.abs(state_vector))
        else:
            # Fallback to classical feature mapping
            quantum_features = X_train
        
        # Convert quantum features to numpy array
        quantum_features = np.array(quantum_features)
        
        # Train classical model on quantum-mapped features
        history = self.classical_model.fit(
            quantum_features, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2
        )
        
        return history
    
    def predict(self, X_test):
        """
        Make predictions using the hybrid model
        
        Parameters:
        - X_test: Test input data
        
        Returns:
        - Predictions
        """
        # Quantum feature mapping for test data
        quantum_features = []
        
        if QUANTUM_AVAILABLE:
            for sample in X_test:
                qc = self.quantum_feature_map(sample)
                # Simulate quantum circuit
                job = execute(qc, self.quantum_backend)
                result = job.result()
                state_vector = result.get_statevector()
                quantum_features.append(np.abs(state_vector))
        else:
            # Fallback to classical feature mapping
            quantum_features = X_test
        
        # Convert quantum features to numpy array
        quantum_features = np.array(quantum_features)
        
        # Predict using classical model
        return self.classical_model.predict(quantum_features)
    
    def visualize_training(self, history):
        """
        Visualize training history
        
        Parameters:
        - history: Training history from model.fit()
        """
        plt.figure(figsize=(16, 5))
        
        # Plot training & validation accuracy values
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot training & validation loss values
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Learning Rate Progression (if available)
        plt.subplot(1, 3, 3)
        plt.plot([1/np.log(i+2) for i in range(len(history.history['loss']))], label='Learning Rate Proxy')
        plt.title('Learning Rate Proxy')
        plt.ylabel('Approximate Learning Rate')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_quantum_features(self, X_train, y_train):
        """
        Visualize quantum feature space using t-SNE
        
        Parameters:
        - X_train: Training input data
        - y_train: Training labels
        """
        # Generate quantum features
        quantum_features = []
        
        if QUANTUM_AVAILABLE:
            for sample in X_train:
                qc = self.quantum_feature_map(sample)
                # Simulate quantum circuit
                job = execute(qc, self.quantum_backend)
                result = job.result()
                state_vector = result.get_statevector()
                quantum_features.append(np.abs(state_vector))
        else:
            # Fallback to classical feature
            quantum_features = X_train
        
        # Convert to numpy array
        quantum_features = np.array(quantum_features)
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        quantum_features_2d = tsne.fit_transform(quantum_features)
        
        # Visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(quantum_features_2d[:, 0], 
                               quantum_features_2d[:, 1], 
                               c=y_train, 
                               cmap='viridis', 
                               alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Quantum Feature Space Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('Quantum Feature Space Mapped with t-SNE')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    
    def quantum_circuit_visualization(self, sample_input):
        """
        Visualize the quantum circuit for a sample input
        
        Parameters:
        - sample_input: A sample input to create quantum circuit
        """
        if not QUANTUM_AVAILABLE:
            print("Quantum circuit visualization not available. Qiskit not fully installed.")
            return
        
        # Create quantum circuit
        qc = self.quantum_feature_map(sample_input)
        
        # Draw the circuit
        plt.figure(figsize=(15, 5))
        plt.title('Quantum Circuit Representation')
        plt.axis('off')
        plt.text(0.5, 0.5, str(qc), ha='center', va='center', fontsize=10, family='monospace')
        plt.show()

if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    X_train = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y_train = (X_train.sum(axis=1) > 5).astype(int)  # Simple binary classification
    
    # Create and train the quantum-inspired neural network
    qinn = QuantumInspiredNeuralNetwork(input_dim=10)
    training_history = qinn.train(X_train, y_train)
    
    # Visualizations
    # 1. Training Progress Visualization
    qinn.visualize_training(training_history)
    
    # 2. Quantum Feature Space Visualization
    qinn.visualize_quantum_features(X_train, y_train)
    
    # 3. Quantum Circuit Visualization (using a sample input)
    if QUANTUM_AVAILABLE:
        qinn.quantum_circuit_visualization(X_train[0])
    else:
        print("Quantum circuit visualization skipped. Qiskit not fully installed.")