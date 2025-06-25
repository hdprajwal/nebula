import os
import sys
import json
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.utils import shuffle
from torch import cuda
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss

logging.basicConfig(level=logging.INFO)

# correct path to repository root
if os.path.basename(os.getcwd()) == "scripts":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.misc import fix_random_seed
from nebula.models import TransformerEncoderChunks
from nebula import ModelTrainer

fix_random_seed(0)


def load_data(tokenizer_type: str = "BPE_50k", vocab_size: int = 50000, maxlen: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], int]:
    """
    Load training and test data for the specified tokenizer.
    
    Args:
        tokenizer_type: Type of tokenizer (e.g., "BPE_50k", "whitespace", etc.)
        vocab_size: Vocabulary size for the tokenizer
        maxlen: Maximum sequence length
        
    Returns:
        Tuple of (xTrain, yTrain, xTest, yTest, vocab, vocab_size)
    """
    logging.info(f" [*] Loading data for {tokenizer_type} tokenizer...")
    
    # Construct folder names based on tokenizer type
    train_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", f"speakeasy_trainset_{tokenizer_type}")
    test_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", f"speakeasy_testset_{tokenizer_type}")
    
    # Load training data
    xTrainFile = os.path.join(train_folder, f"speakeasy_vocab_size_{vocab_size}_maxlen_{maxlen}_x.npy")
    xTrain = np.load(xTrainFile)
    yTrainFile = os.path.join(train_folder, "speakeasy_y.npy")
    yTrain = np.load(yTrainFile)
    
    # Load test data
    xTestFile = os.path.join(test_folder, f"speakeasy_vocab_size_{vocab_size}_maxlen_{maxlen}_x.npy")
    xTest = np.load(xTestFile)
    yTestFile = os.path.join(test_folder, "speakeasy_y.npy")
    yTest = np.load(yTestFile)
    
    # Load vocabulary
    vocabFile = os.path.join(train_folder, f"speakeasy_vocab_size_{vocab_size}_tokenizer_vocab.json")
    with open(vocabFile, 'r') as f:
        vocab = json.load(f)
    
    actual_vocab_size = len(vocab)
    logging.info(f" [*] Loaded data. Vocab size: {actual_vocab_size}")
    
    return xTrain, yTrain, xTest, yTest, vocab, actual_vocab_size


def create_model_config(vocab_size: int, maxlen: int = 512, **kwargs) -> Dict[str, Any]:
    """
    Create model configuration dictionary.
    
    Args:
        vocab_size: Size of the vocabulary
        maxlen: Maximum sequence length
        **kwargs: Additional model configuration parameters to override defaults
        
    Returns:
        Model configuration dictionary
    """
    default_config = {
        "vocab_size": vocab_size,
        "maxlen": maxlen,
        "chunk_size": 64,  # input splitting to chunks
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": 1,  # binary classification
        "classifier_head": [64],  # classifier ffnn dims
        "layerNorm": False,
        "dropout": 0.3,
        "norm_first": True
    }
    
    # Override with any provided kwargs
    default_config.update(kwargs)
    return default_config


def create_model(model_config: Dict[str, Any], pretrained_model_path: Optional[str] = None) -> TransformerEncoderChunks:
    """
    Create and optionally load a pre-trained model.
    
    Args:
        model_config: Model configuration dictionary
        pretrained_model_path: Path to pre-trained model weights (optional)
        
    Returns:
        Initialized model
    """
    logging.info(f" [*] Creating model...")
    
    model = TransformerEncoderChunks(**model_config)
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        logging.info(f" [*] Loading pre-trained weights from {pretrained_model_path}")
        from torch import load
        state_dict = load(pretrained_model_path, weights_only=False)
        # Clean up from LM layers if present
        state_dict = {k: v for k, v in state_dict.items() if "pretrain_layers" not in k}
        model.load_state_dict(state_dict)
    
    logging.info(f" [!] Model ready.")
    return model


def create_trainer_config(model: TransformerEncoderChunks, time_budget: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """
    Create model trainer configuration.
    
    Args:
        model: The model to train
        time_budget: Training time budget in minutes (optional)
        **kwargs: Additional trainer configuration parameters to override defaults
        
    Returns:
        Trainer configuration dictionary
    """
    device = "cuda" if cuda.is_available() else "cpu"
    
    default_config = {
        "device": device,
        "model": model,
        "loss_function": BCEWithLogitsLoss(),
        "optimizer_class": AdamW,
        "optimizer_config": {"lr": 3e-4},
        "optim_scheduler": None,
        "optim_step_budget": None,
        "outputFolder": "bpe_50k_out",
        "batchSize": 96,
        "verbosity_n_batches": 100,
        "clip_grad_norm": 1.0,
        "n_batches_grad_update": 1,
        "time_budget": int(time_budget * 60) if time_budget else None,
    }
    
    # Override with any provided kwargs
    default_config.update(kwargs)
    return default_config


def train_model(xTrain: np.ndarray, yTrain: np.ndarray, 
                tokenizer_type: str = "BPE_50k", 
                vocab_size: int = 50000, 
                maxlen: int = 512,
                time_budget: Optional[int] = 5,
                pretrained_model_path: Optional[str] = None,
                model_config_overrides: Optional[Dict[str, Any]] = None,
                trainer_config_overrides: Optional[Dict[str, Any]] = None) -> ModelTrainer:
    """
    Complete training pipeline.
    
    Args:
        xTrain: Training input data
        yTrain: Training labels
        tokenizer_type: Type of tokenizer used
        vocab_size: Vocabulary size
        maxlen: Maximum sequence length
        time_budget: Training time budget in minutes
        pretrained_model_path: Path to pre-trained model weights
        model_config_overrides: Dictionary to override default model config
        trainer_config_overrides: Dictionary to override default trainer config
        
    Returns:
        Trained ModelTrainer instance
    """
    # Create model configuration
    model_config = create_model_config(
        vocab_size=vocab_size, 
        maxlen=maxlen, 
        **(model_config_overrides or {})
    )
    
    # Create model
    model = create_model(model_config, pretrained_model_path)
    
    # Create trainer configuration
    trainer_config = create_trainer_config(
        model=model, 
        time_budget=time_budget, 
        **(trainer_config_overrides or {})
    )
    
    # Create and run trainer
    model_trainer = ModelTrainer(**trainer_config)
    model_trainer.train(xTrain, yTrain)

    return model_trainer


def evaluate_model(model_trainer: ModelTrainer, xTest: np.ndarray, yTest: np.ndarray):
    results = model_trainer.evaluate(xTest, yTest, metrics="json")
    with open("results.json", "w") as f:
        json.dump(results, f)
    logging.info(f" [*] Evaluation results: {results}")
    return results

def main():
    """Main training function that can be easily configured for different tokenizers."""
    
    # Configuration
    TOKENIZER_TYPE = "BPE_50k"  # Change this to "whitespace" or other tokenizer types
    VOCAB_SIZE = 50000
    MAXLEN = 512
    TIME_BUDGET = 5  # minutes
    
    # Load data
    xTrain, yTrain, xTest, yTest, vocab, actual_vocab_size = load_data(
        tokenizer_type=TOKENIZER_TYPE,
        vocab_size=VOCAB_SIZE,
        maxlen=MAXLEN
    )
    
    # Optional: specify pretrained model path
    # pretrained_model_path = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", "speakeasy_vocab_size_50000_tokenizer.model" )
    pretrained_model_path = None    

    # Optional: override model configuration
    model_config_overrides = {}
    
    # Optional: override trainer configuration
    trainer_config_overrides = {}
    
    # Train the model
    model_trainer = train_model(
        xTrain=xTrain,
        yTrain=yTrain,
        tokenizer_type=TOKENIZER_TYPE,
        vocab_size=actual_vocab_size,
        maxlen=MAXLEN,
        time_budget=TIME_BUDGET,
        pretrained_model_path=pretrained_model_path,
        model_config_overrides=model_config_overrides,
        trainer_config_overrides=trainer_config_overrides
    )
    
    logging.info(" [!] Training completed!")

    results = evaluate_model(model_trainer, xTest, yTest)
    
    return model_trainer, results


if __name__ == "__main__":
    main()