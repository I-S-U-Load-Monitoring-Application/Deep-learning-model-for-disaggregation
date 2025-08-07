"""
Model Signature Components for App-Ready NILM Models
====================================================

This module demonstrates how to add proper signatures to make your NILM model
production-ready and easily integratable into applications.
"""

import tensorflow as tf
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelMetadata:
    """
    Model metadata signature - Contains all essential information about the model.
    """
    model_name: str
    version: str
    appliance_name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    window_length: int
    sampling_rate: str  # e.g., "8S" for 8 seconds
    
    # Normalization parameters (CRITICAL for app integration)
    aggregate_mean: float
    aggregate_std: float
    appliance_mean: float
    appliance_std: float
    
    # Model performance metrics
    training_loss: float
    validation_loss: float
    mae: float
    
    # Training configuration
    epochs_trained: int
    batch_size: int
    learning_rate: float
    
    # Data information
    training_houses: List[int]
    test_house: int
    dataset_name: str
    
    # Timestamps
    created_at: str
    trained_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelMetadata':
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class AppReadyNILMModel:
    """
    App-ready wrapper for NILM models with complete signature.
    
    This class provides a standardized interface for using NILM models
    in production applications.
    """
    
    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Initialize the app-ready model.
        
        Args:
            model_path: Path to the saved Keras model
            metadata_path: Path to the metadata JSON file
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.model_path.with_suffix('.json')
        
        # Load model and metadata
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        
        # Extract commonly used parameters
        self.window_length = self.metadata.window_length
        self.window_offset = (self.window_length - 1) // 2
        
    def _load_model(self) -> tf.keras.Model:
        """Load the Keras model with error handling."""
        try:
            return tf.keras.models.load_model(str(self.model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def _load_metadata(self) -> ModelMetadata:
        """Load model metadata with fallback."""
        if self.metadata_path.exists():
            return ModelMetadata.load(str(self.metadata_path))
        else:
            # Create minimal metadata if not found
            return self._create_minimal_metadata()
    
    def _create_minimal_metadata(self) -> ModelMetadata:
        """Create minimal metadata from model inspection."""
        input_shape = self.model.input_shape[1:]  # Remove batch dimension
        output_shape = self.model.output_shape[1:]  # Remove batch dimension
        
        return ModelMetadata(
            model_name=self.model_path.stem,
            version="1.0",
            appliance_name="unknown",
            input_shape=input_shape,
            output_shape=output_shape,
            window_length=input_shape[0],
            sampling_rate="8S",
            aggregate_mean=522.0,  # UK-DALE defaults
            aggregate_std=814.0,
            appliance_mean=0.0,
            appliance_std=1.0,
            training_loss=0.0,
            validation_loss=0.0,
            mae=0.0,
            epochs_trained=0,
            batch_size=0,
            learning_rate=0.0,
            training_houses=[],
            test_house=0,
            dataset_name="unknown",
            created_at="unknown",
            trained_at="unknown"
        )
    
    def preprocess_input(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Preprocess raw aggregate power data for model input.
        
        Args:
            raw_data: Raw aggregate power readings (1D array)
            
        Returns:
            Preprocessed data ready for model inference
        """
        # Normalize using stored parameters
        normalized_data = (raw_data - self.metadata.aggregate_mean) / self.metadata.aggregate_std
        
        # Create sliding windows
        if len(normalized_data) < self.window_length:
            raise ValueError(f"Input data length ({len(normalized_data)}) must be >= window length ({self.window_length})")
        
        windows = []
        for i in range(len(normalized_data) - self.window_length + 1):
            window = normalized_data[i:i + self.window_length]
            windows.append(window)
        
        # Reshape for model: (num_windows, window_length, 1)
        return np.array(windows).reshape(-1, self.window_length, 1)
    
    def postprocess_output(self, model_output: np.ndarray) -> np.ndarray:
        """
        Postprocess model output to get actual power values.
        
        Args:
            model_output: Raw model predictions
            
        Returns:
            Denormalized appliance power predictions
        """
        # Denormalize using stored parameters
        denormalized = (model_output * self.metadata.appliance_std) + self.metadata.appliance_mean
        
        # Ensure non-negative values (power cannot be negative)
        return np.maximum(denormalized, 0)
    
    def predict(self, raw_data: np.ndarray) -> Dict[str, Any]:
        """
        Complete prediction pipeline with preprocessing and postprocessing.
        
        Args:
            raw_data: Raw aggregate power readings
            
        Returns:
            Dictionary containing predictions and metadata
        """
        try:
            # Preprocess
            processed_input = self.preprocess_input(raw_data)
            
            # Predict
            raw_predictions = self.model.predict(processed_input, verbose=0)
            
            # Postprocess
            final_predictions = self.postprocess_output(raw_predictions).flatten()
            
            return {
                'predictions': final_predictions.tolist(),
                'num_predictions': len(final_predictions),
                'appliance': self.metadata.appliance_name,
                'confidence': 'high',  # You can add confidence scoring
                'preprocessing': {
                    'aggregate_mean': self.metadata.aggregate_mean,
                    'aggregate_std': self.metadata.aggregate_std,
                    'window_length': self.window_length
                },
                'model_info': {
                    'version': self.metadata.version,
                    'training_houses': self.metadata.training_houses
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def predict_batch(self, batch_data: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Predict on a batch of data samples.
        
        Args:
            batch_data: List of raw aggregate power arrays
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(data) for data in batch_data]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'metadata': self.metadata.to_dict(),
            'model_summary': {
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.utils.count_params(w) for w in self.model.trainable_weights]),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape
            },
            'signature': {
                'input_spec': {
                    'shape': self.metadata.input_shape,
                    'dtype': 'float32',
                    'description': 'Normalized aggregate power windows'
                },
                'output_spec': {
                    'shape': self.metadata.output_shape,
                    'dtype': 'float32',
                    'description': 'Predicted appliance power consumption'
                }
            }
        }


class ModelSigner:
    """
    Utility class to add signatures to trained models.
    """
    
    @staticmethod
    def sign_model(
        model: tf.keras.Model,
        model_path: str,
        appliance_name: str,
        training_config: Dict[str, Any],
        training_history: tf.keras.callbacks.History,
        normalization_params: Dict[str, float]
    ) -> AppReadyNILMModel:
        """
        Add complete signature to a trained model.
        
        Args:
            model: Trained Keras model
            model_path: Where to save the signed model
            appliance_name: Target appliance name
            training_config: Training configuration used
            training_history: Training history from model.fit()
            normalization_params: Normalization parameters used
            
        Returns:
            App-ready model instance
        """
        from datetime import datetime
        
        # Save the model
        model.save(model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=f"{appliance_name}_seq2point",
            version="1.0",
            appliance_name=appliance_name,
            input_shape=model.input_shape[1:],  # Remove batch dimension
            output_shape=model.output_shape[1:],
            window_length=model.input_shape[1],
            sampling_rate="8S",
            
            # Normalization parameters (CRITICAL!)
            aggregate_mean=normalization_params['aggregate_mean'],
            aggregate_std=normalization_params['aggregate_std'],
            appliance_mean=normalization_params['appliance_mean'],
            appliance_std=normalization_params['appliance_std'],
            
            # Performance metrics
            training_loss=float(min(training_history.history['loss'])),
            validation_loss=float(min(training_history.history.get('val_loss', [0]))),
            mae=float(min(training_history.history.get('mae', [0]))),
            
            # Training info
            epochs_trained=len(training_history.history['loss']),
            batch_size=training_config.get('batch_size', 0),
            learning_rate=training_config.get('learning_rate', 0.001),
            
            # Data info
            training_houses=training_config.get('training_houses', []),
            test_house=training_config.get('test_house', 0),
            dataset_name=training_config.get('dataset_name', 'UK-DALE'),
            
            # Timestamps
            created_at=datetime.now().isoformat(),
            trained_at=datetime.now().isoformat()
        )
        
        # Save metadata
        metadata_path = Path(model_path).with_suffix('.json')
        metadata.save(str(metadata_path))
        
        # Return app-ready model
        return AppReadyNILMModel(model_path, str(metadata_path))


def create_tensorflow_serving_signature(model: tf.keras.Model, export_path: str) -> None:
    """
    Create TensorFlow Serving signature for deployment.
    
    Args:
        model: Trained Keras model
        export_path: Where to export the serving model
    """
    # Define the signature
    @tf.function
    def serving_fn(input_data):
        return {'predictions': model(input_data)}
    
    # Get input signature
    input_signature = tf.TensorSpec(
        shape=(None, model.input_shape[1], model.input_shape[2]), 
        dtype=tf.float32
    )
    
    # Save for TensorFlow Serving
    tf.saved_model.save(
        model,
        export_path,
        signatures={'serving_default': serving_fn.get_concrete_function(input_signature)}
    )


if __name__ == "__main__":
    print("Model Signature Guide for App Integration")
    print("=========================================")
    print("\nThis module shows how to add proper signatures to your NILM models")
    print("for production deployment and app integration.")