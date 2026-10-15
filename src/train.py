import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (set_seed, load_cpi_data, split_data,
                        generate_esm2_embeddings, BiCrossAttCPIDataset,
                        create_dataloaders)
from models import BiCrossAttCPI


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, v, n=1):
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-12)


def evaluate(model, loader, device, criterion=None):
    """Evaluate model on a DataLoader."""
    model.eval()
    loss_m = AverageMeter()
    preds_all, labs_all, probs_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch,
                           batch.protein_embedding)
            if criterion is not None:
                loss_m.update(criterion(logits, batch.y.view(-1)).item(),
                              batch.y.size(0))
            probs = torch.softmax(logits, 1)
            preds_all.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
            labs_all.extend(batch.y.view(-1).cpu().numpy().tolist())
            probs_all.extend(probs[:, 1].cpu().numpy().tolist())

    labs_all = np.array(labs_all).flatten()
    preds_all = np.array(preds_all).flatten()
    acc = accuracy_score(labs_all, preds_all)
    prec = precision_score(labs_all, preds_all, zero_division=0)
    rec = recall_score(labs_all, preds_all, zero_division=0)
    f1 = f1_score(labs_all, preds_all, zero_division=0)
    try:
        auc_v = roc_auc_score(labs_all, probs_all)
    except Exception:
        auc_v = 0.0
    if criterion is not None:
        return loss_m.avg, acc, prec, rec, f1, auc_v, labs_all, preds_all, probs_all
    return acc, prec, rec, f1, auc_v, labs_all, preds_all, probs_all


def train_epoch(model, loader, opt, crit, device):
    """Train for one epoch."""
    model.train()
    loss_m = AverageMeter()
    P, L = [], []
    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch,
                       batch.protein_embedding)
        loss = crit(logits, batch.y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_m.update(loss.item(), batch.y.size(0))
        P.extend(torch.argmax(logits, 1).cpu().numpy().tolist())
        L.extend(batch.y.view(-1).cpu().numpy().tolist())
    return loss_m.avg, accuracy_score(np.array(L).flatten(), np.array(P).flatten())


def train_model(model, train_loader, val_loader, opt, crit, device,
                epochs=100, model_name="BiCrossAtt-CPI", dataset_name="Human",
                save_path=None, verbose=True):
    """Train for the full epoch count without early stopping."""
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_state = None

    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [], 'val_auc': []}

    if save_path is None:
        save_path = f'best_model_{model_name}_{dataset_name}.pt'

    if verbose:
        print("=" * 100)
        print(f"{'Ep':<5}{'Time':<10}{'TrLoss':<12}{'TrAcc':<12}"
              f"{'VaLoss':<12}{'VaAcc':<12}{'VaAUC':<12}")
        print("=" * 100)

    for ep in range(epochs):
        t0 = time.time()
        trl, tra = train_epoch(model, train_loader, opt, crit, device)
        val_loss, va, vp, vr, vf, vau, _, _, _ = evaluate(
            model, val_loader, device, crit)
        dt = time.time() - t0

        history['train_loss'].append(trl)
        history['train_acc'].append(tra)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(va)
        history['val_auc'].append(vau)

        if verbose:
            print(f"{ep+1:<5}{dt:<10.2f}{trl:<12.4f}{tra:<12.4f}"
                  f"{val_loss:<12.4f}{va:<12.4f}{vau:<12.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = va
            best_val_auc = vau
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                'model_state_dict': best_state,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'val_auc': best_val_auc,
                'epoch': ep + 1
            }, save_path)

    if best_state is not None:
        print(f"\n✅ Training finished. Restoring best model "
              f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, "
              f"val_auc={best_val_auc:.4f})")
        model.load_state_dict(best_state)

    return model, history, best_val_loss, best_val_acc, best_val_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train BiCrossAtt-CPI")
    parser.add_argument('--data_path', type=str, default='data/data.txt',
                        help='Path to data file')
    parser.add_argument('--dataset', type=str, default='Human',
                        choices=['Human', 'C. elegans'],
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--esm_model', type=str, default='esm2_t30_150M_UR50D')
    parser.add_argument('--esm_batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    df = load_cpi_data(args.data_path)
    train_df, val_df, test_df = split_data(df, seed=args.seed)

    os.makedirs('data/raw', exist_ok=True)
    train_df.to_csv('data/raw/data_train.csv', index=False)
    val_df.to_csv('data/raw/data_val.csv', index=False)
    test_df.to_csv('data/raw/data_test.csv', index=False)

    # --- Generate/load ESM-2 embeddings ---
    cache = f'esm2_embeddings_cache_{args.dataset.replace(" ", "_")}.pt'
    all_seqs = (set(train_df.target_sequence) |
                set(val_df.target_sequence) |
                set(test_df.target_sequence))
    print(f"Unique sequences: {len(all_seqs)}")

    if os.path.exists(cache):
        protein_embeddings = torch.load(cache, weights_only=False)
        print("Loaded cached embeddings")
    else:
        protein_embeddings = generate_esm2_embeddings(
            list(all_seqs), batch_size=args.esm_batch_size,
            model_name=args.esm_model, device='cpu')
        torch.save(protein_embeddings, cache)
        print("Saved embeddings cache")

    protein_dim = list(protein_embeddings.values())[0].shape[0]
    print(f"Protein embedding dim: {protein_dim}")

    # --- Build datasets ---
    train_ds = BiCrossAttCPIDataset('./data', types='train',
                                    protein_embeddings=protein_embeddings,
                                    protein_dim=protein_dim)
    val_ds = BiCrossAttCPIDataset('./data', types='val',
                                  protein_embeddings=protein_embeddings,
                                  protein_dim=protein_dim)
    test_ds = BiCrossAttCPIDataset('./data', types='test',
                                   protein_embeddings=protein_embeddings,
                                   protein_dim=protein_dim)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size)

    # --- Build model ---
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
        dropout_rate=args.dropout
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # --- Train ---
    save_path = os.path.join(args.output_dir,
                             f'best_model_BiCrossAtt-CPI_{args.dataset}.pt')
    model, history, bvl, bva, bau = train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=args.epochs, model_name="BiCrossAtt-CPI",
        dataset_name=args.dataset, save_path=save_path, verbose=True)

    # --- Test evaluation ---
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, _, _, _ = evaluate(
        model, test_loader, device, criterion)
    print("\n" + "=" * 60)
    print(f"🎯 Test Results — BiCrossAtt-CPI ({args.dataset}):")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall   : {test_rec:.4f}")
    print(f"  F1       : {test_f1:.4f}")
    print(f"  AUC      : {test_auc:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
