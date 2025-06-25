import os
import sys
import json
from time import time

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from sklearn.utils import shuffle

from torch import load
# correct path to repository root
if os.path.basename(os.getcwd()) == "scripts":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.evaluation.lit_cv import LitCrossValidation
from nebula.models import TransformerEncoderChunks
from nebula.misc import fix_random_seed

if __name__ == "__main__":
    fix_random_seed(0)
    TRAINING = False
    # ==============
    # LOAD DATA
    # ==============
    logging.info(f" [*] Loading data...")
    train_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k")
    xTrainFile = os.path.join(train_folder, f"speakeasy_vocab_size_50000_maxlen_512_x.npy")
    x_train = np.load(xTrainFile)
    yTrainFile = os.path.join(train_folder, "speakeasy_y.npy")
    y_train = np.load(yTrainFile)
    test_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", "speakeasy_testset_BPE_50k")
    xTestFile = os.path.join(test_folder, f"speakeasy_vocab_size_50000_maxlen_512_x.npy")
    x_test = np.load(xTestFile)
    yTestFile = os.path.join(test_folder, "speakeasy_y.npy")
    y_test = np.load(yTestFile)

    # shuffle and limit
    limit = None
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_train = x_train[:limit]
    y_train = y_train[:limit]
    x_test, y_test = shuffle(x_test, y_test, random_state=0)
    x_test = x_test[:limit]
    y_test = y_test[:limit]

    vocabFile = os.path.join(train_folder, r"speakeasy_vocab_size_50000_tokenizer_vocab.json")
    with open(vocabFile, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    logging.info(f" [*] Loaded data.")

    # ===================
    # MODELING
    # ===================
    logging.info(f" [*] Loading model...")
    model_config = {
        "vocab_size": vocab_size,
        "maxlen": 512,
        "chunk_size": 64, # input splitting to chunks
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": 1, # binary classification
        "classifier_head": [64], # classifier ffnn dims
        "layerNorm": False,
        "dropout": 0.3,
        "norm_first": True
    }
    model = TransformerEncoderChunks(**model_config)
    logging.info(f" [!] Model ready.")


    lit_cv = LitCrossValidation(
        # cv config
        folds=3,
        dump_data_splits=True,
        dump_models=True,
        # trainer config
        pytorch_model=model,
        name="test_training",
        log_folder=f"./cv_test_run_{int(time())}",
        epochs=3,
        scheduler="onecycle",
        # data config
        batch_size=256,
        dataloader_workers=4,
        # misc
        random_state=0,
        verbose=True,
        device="gpu"
    )
    lit_cv.run(x=x_train, y=y_train, print_fold_scores=True)

    logging.info(f" [*] Evaluating on test set...")

    # best_model_path = os.path.join(lit_cv.log_folder )
    
    # if not os.path.exists(best_model_path):
    #     print(f"Model file not found: {best_model_path}")
    #     available_files = os.listdir(lit_cv.log_folder)
    #     ckpt_files = [f for f in available_files if f.endswith('.ckpt')]
    #     print(f"Available .ckpt files: {ckpt_files}")
    #     exit(1)
    
    # print(f"[*] Loading model from {best_model_path}...")
    # lit_cv.load_lit_model(best_model_path)

    test_loader = lit_cv.create_dataloader(x_test, y_test, shuffle=False)
    y_pred = lit_cv.predict_lit_model(test_loader, return_logits=False, decision_threshold=0.5)
    y_pred_proba = lit_cv.predict_lit_model(test_loader, return_logits=True)

    # Calculate basic metrics
    from sklearn.metrics import f1_score, roc_auc_score, classification_report, roc_curve
    import numpy as np
    
    f1_test = f1_score(y_test, y_pred)
    auc_test = roc_auc_score(y_test, y_pred_proba)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    print(f"fpr: {fpr}")
    print(f"tpr: {tpr}")
    print(f"thresholds: {thresholds}")

    # Calculate TPR at specific FPR values (including 10^-3)
    target_fprs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]  # Including 10^-3
    
    print(f"\n{'='*50}")
    print(f"TEST SET EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall F1 Score: {f1_test:.4f}")
    print(f"Overall AUC-ROC:  {auc_test:.4f}")
    print(f"Test set size:    {len(y_test)}")
    
    print(f"\nTPR at specific FPR thresholds:")
    print(f"{'FPR':<10} {'TPR':<8} {'F1':<8} {'Threshold':<10}")
    print(f"{'-'*40}")
    
    for target_fpr in target_fprs:
        if np.any(fpr <= target_fpr):
            tpr_at_fpr = tpr[fpr <= target_fpr][-1]
            threshold_at_fpr = thresholds[fpr <= target_fpr][-1]
            
            # Calculate F1 at this threshold
            y_pred_at_threshold = (y_pred_proba >= threshold_at_fpr).astype(int) if isinstance(y_pred_proba, np.ndarray) else (y_pred_proba >= threshold_at_fpr).int().numpy()
            f1_at_fpr = f1_score(y_test, y_pred_at_threshold)
            
            print(f"{target_fpr:<10.1e} {tpr_at_fpr:<8.4f} {f1_at_fpr:<8.4f} {threshold_at_fpr:<10.4f}")
        else:
            print(f"{target_fpr:<10.1e} {'N/A':<8} {'N/A':<8} {'N/A':<10}")
    
    # Specifically highlight TPR at FPR = 10^-3
    target_fpr_001 = 0.001
    if np.any(fpr <= target_fpr_001):
        tpr_at_001 = tpr[fpr <= target_fpr_001][-1]
        print(f"\n🎯 ANSWER: TPR at FPR = 10^-3 (0.001) = {tpr_at_001:.4f}")
    else:
        print(f"\n❌ Cannot achieve FPR = 10^-3 with current model predictions")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
