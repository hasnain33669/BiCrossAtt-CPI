import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_curve, auc)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (set_seed, load_cpi_data, split_data,
                        generate_esm2_embeddings, BiCrossAttCPIDataset,
                        create_dataloaders)
from models import BiCrossAttCPI


def evaluate_full(model, loader, device, criterion=None):
    """Evaluate model and return all metrics + raw predictions."""
    model.eval()
    preds_all, labs_all, probs_all = [], [], []
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch,
                           batch.protein_embedding)
            if criterion is not None:
                total_loss += criterion(logits, batch.y.view(-1)).item() * batch.y.size(0)
                n_samples += batch.y.size(0)
            probs = torch.softmax(logits, 1)
            preds_all.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
            labs_all.extend(batch.y.view(-1).cpu().numpy().tolist())
            probs_all.extend(probs[:, 1].cpu().numpy().tolist())

    labs_all = np.array(labs_all).flatten()
    preds_all = np.array(preds_all).flatten()
    probs_all = np.array(probs_all).flatten()

    metrics = {
        'loss': total_loss / max(n_samples, 1),
        'accuracy': accuracy_score(labs_all, preds_all),
        'precision': precision_score(labs_all, preds_all, zero_division=0),
        'recall': recall_score(labs_all, preds_all, zero_division=0),
        'f1': f1_score(labs_all, preds_all, zero_division=0),
        'auc': roc_auc_score(labs_all, probs_all) if len(np.unique(labs_all)) > 1 else 0.0,
    }

    # Confusion matrix
    cm = confusion_matrix(labs_all, preds_all)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        })

    return metrics, labs_all, preds_all, probs_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiCrossAtt-CPI")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/data.txt')
    parser.add_argument('--dataset', type=str, default='Human',
                        choices=['Human', 'C. elegans'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--esm_model', type=str, default='esm2_t30_150M_UR50D')
    parser.add_argument('--esm_batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load data ---
    df = load_cpi_data(args.data_path)
    train_df, val_df, test_df = split_data(df, seed=args.seed)

    # --- Embeddings ---
    cache = f'esm2_embeddings_cache_{args.dataset.replace(" ", "_")}.pt'
    all_seqs = (set(train_df.target_sequence) |
                set(val_df.target_sequence) |
                set(test_df.target_sequence))

    if os.path.exists(cache):
        protein_embeddings = torch.load(cache, weights_only=False)
        print("Loaded cached embeddings")
    else:
        protein_embeddings = generate_esm2_embeddings(
            list(all_seqs), batch_size=args.esm_batch_size,
            model_name=args.esm_model, device='cpu')
        torch.save(protein_embeddings, cache)

    protein_dim = list(protein_embeddings.values())[0].shape[0]

    # --- Datasets ---
    test_ds = BiCrossAttCPIDataset('./data', types='test',
                                   protein_embeddings=protein_embeddings,
                                   protein_dim=protein_dim)
    _, _, test_loader = create_dataloaders(
        test_ds, test_ds, test_ds, batch_size=args.batch_size)

    # --- Model ---
    model = BiCrossAttCPI(
        compound_input_dim=87,
        protein_embed_dim=protein_dim,
        cross_attn_heads=8,
        cross_attn_dropout=0.1,
        num_classes=2,
        use_compound_gnn=True,
        use_cross_attention=True,
        use_c2p_attention=True,
        use_p2c_attention=True,
        dropout_rate=0.2
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.model_path}")
    print(f"Best val_loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Best val_acc:  {checkpoint.get('val_acc', 'N/A')}")
    print(f"Best val_auc:  {checkpoint.get('val_auc', 'N/A')}")

    # --- Evaluate ---
    criterion = nn.CrossEntropyLoss()
    metrics, labs, preds, probs = evaluate_full(model, test_loader, device, criterion)

    print("\n" + "=" * 60)
    print(f"📊 Test Results — BiCrossAtt-CPI ({args.dataset})")
    print("=" * 60)
    print(f"  Loss        : {metrics['loss']:.4f}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  F1-Score    : {metrics['f1']:.4f}")
    print(f"  AUC-ROC     : {metrics['auc']:.4f}")
    print("-" * 60)
    print(f"  Confusion Matrix:")
    print(f"    TN={metrics['tn']}  FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}  TP={metrics['tp']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
