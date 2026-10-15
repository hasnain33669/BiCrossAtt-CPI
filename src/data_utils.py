"""
Data utilities for BiCrossAtt-CPI.

"""

import os
import os.path as osp
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

import rdkit
from rdkit import Chem
from rdkit import rdBase

# Suppress RDKit warnings
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cpi_data(data_path='data/data.txt'):
    """
    Load CPI data from a text file.

    Expected format: <SMILES> <Protein_Sequence> <Label>

    Returns:
        pd.DataFrame with columns: compound_iso_smiles, target_sequence, affinity
    """
    print("=" * 60)
    print("LOADING CPI DATA")
    print("=" * 60)

    if not os.path.exists(data_path):
        # Try alternate paths
        for alt in ['/content/data.txt', 'data.txt', '../data/data.txt']:
            if os.path.exists(alt):
                data_path = alt
                break
        else:
            raise FileNotFoundError(
                f"Data file not found at {data_path}. "
                f"Please place data.txt in the data/ directory."
            )

    print(f"Loading data from: {data_path}")

    smiles_list, protein_list, label_list = [], [], []

    with open(data_path, 'r') as f:
        lines = f.readlines()

    print(f"Total lines in file: {len(lines)}")

    invalid_count = 0
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        if len(parts) >= 3:
            smiles, protein, label = parts[0], parts[1], parts[2]

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles_list.append(smiles)
                    protein_list.append(protein)
                    label_list.append(int(label))
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1
        else:
            invalid_count += 1

    df = pd.DataFrame({
        'compound_iso_smiles': smiles_list,
        'target_sequence': protein_list,
        'affinity': label_list
    })

    print(f"\n✅ Loaded {len(df)} valid samples")
    print(f"❌ Skipped {invalid_count}")
    print(f"Positive: {df['affinity'].sum()} | Negative: {len(df) - df['affinity'].sum()}")

    return df


def split_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split data into train/val/test sets with stratification."""
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        random_state=seed, stratify=df['affinity']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=seed, stratify=temp_df['affinity']
    )
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Atom feature encoding
# ---------------------------------------------------------------------------
def one_of_k_encoding(x, allow):
    if x not in allow:
        raise Exception(f"{x} not in {allow}")
    return [x == s for s in allow]


def one_of_k_encoding_unk(x, allow):
    if x not in allow:
        x = allow[-1]
    return [x == s for s in allow]


def atom_features(atom):
    """Compute atom-level features (87-dimensional)."""
    enc = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    enc += one_of_k_encoding(atom.GetDegree(), list(range(11)))
    enc += one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
    enc += one_of_k_encoding_unk(
        atom.GetValence(Chem.ValenceType.IMPLICIT),
        list(range(11))
    )
    enc += one_of_k_encoding_unk(
        atom.GetHybridization(),
        [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
         Chem.rdchem.HybridizationType.SP3D2, 'other'])
    enc += [atom.GetIsAromatic()]
    try:
        enc += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S'])
        enc += [atom.HasProp('_ChiralityPossible')]
    except Exception:
        enc += [0, 0] + [atom.HasProp('_ChiralityPossible')]
    return np.array(enc, dtype=np.float32)


def mol_to_graph(mol):
    """Convert RDKit molecule to node features and edge index."""
    import networkx as nx

    feats = []
    for a in mol.GetAtoms():
        f = atom_features(a)
        f = f / (np.sum(f) + 1e-8)
        feats.append(f)
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    if len(edges) == 0:
        return feats, [[0, 0]]
    g = nx.Graph(edges).to_directed()
    return feats, [[e1, e2] for e1, e2 in g.edges]


# ---------------------------------------------------------------------------
# ESM-2 embedding generation
# ---------------------------------------------------------------------------
def generate_esm2_embeddings(sequences, batch_size=1, model_name='esm2_t30_150M_UR50D',
                              repr_layer=30, device='cpu'):
    """
    Generate mean-pooled per-residue ESM-2 embeddings for protein sequences.

    Args:
        sequences: iterable of protein sequences
        batch_size: batch size for ESM inference
        model_name: ESM model identifier
        repr_layer: representation layer to extract
        device: torch device

    Returns:
        dict mapping sequence -> embedding tensor (D,)
    """
    import esm

    print(f"Loading ESM-2 ({model_name})...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    bc = alphabet.get_batch_converter()
    model = model.to(torch.device(device)).eval()

    emb_dict = {}
    seqs = list(sequences)
    for i in tqdm(range(0, len(seqs), batch_size), desc="ESM-2"):
        batch_seqs = seqs[i:i + batch_size]
        batch = [(f"p{j}", s) for j, s in enumerate(batch_seqs)]
        _, _, toks = bc(batch)
        with torch.no_grad():
            out = model(toks.to(device), repr_layers=[repr_layer])
            reps = out["representations"][repr_layer]
            for j, s in enumerate(batch_seqs):
                emb_dict[s] = reps[j, 1:len(s) + 1, :].mean(0).cpu()
    return emb_dict


# ---------------------------------------------------------------------------
# PyG Dataset
# ---------------------------------------------------------------------------
class BiCrossAttCPIDataset(InMemoryDataset):
    """PyG dataset that builds Data objects from CSVs + protein embeddings."""

    def __init__(self, root, types='train', protein_embeddings=None,
                 transform=None, pre_transform=None,
                 force_reprocess=False, protein_dim=None):
        self.types = types
        self.protein_embeddings = protein_embeddings
        self.force_reprocess = force_reprocess
        self.protein_dim = protein_dim
        super().__init__(root, transform, pre_transform)
        idx = {'train': 0, 'val': 1, 'test': 2}[types]
        self.data, self.slices = torch.load(
            self.processed_paths[idx], map_location='cpu', weights_only=False
        )

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_val.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_train.pt', 'processed_val.pt', 'processed_test.pt']

    def download(self):
        pass

    def _process(self):
        needs_rebuild = (
            self.force_reprocess
            or not all(osp.exists(p) for p in self.processed_paths)
        )
        if not needs_rebuild:
            return

        for p in self.processed_paths:
            if osp.exists(p):
                os.remove(p)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

    def process_data(self, path, graph_dict):
        d = pd.read_csv(path)
        out = []
        n_missing = 0
        for _, row in d.iterrows():
            smi = row['compound_iso_smiles']
            seq = row['target_sequence']
            lab = row['affinity']

            if smi not in graph_dict:
                continue

            x, ei = graph_dict[smi]
            pe = self.protein_embeddings.get(seq) if self.protein_embeddings else None
            missing = 0
            if pe is None:
                pe = torch.zeros(self.protein_dim)
                missing = 1
                n_missing += 1

            data = Data(
                x=torch.FloatTensor(np.array(x)),
                edge_index=torch.LongTensor(ei).t().contiguous(),
                y=torch.LongTensor([lab]),
                protein_embedding=pe.unsqueeze(0),
                protein_missing=torch.LongTensor([missing]),
                target_sequence=seq,
            )
            out.append(data)

        if n_missing > 0:
            print(f"   ⚠️  [{self.types}] {n_missing} rows had no embedding "
                  f"-> used zero vector")
        return out

    def process(self):
        dtr = pd.read_csv(self.raw_paths[0])
        dva = pd.read_csv(self.raw_paths[1])
        dte = pd.read_csv(self.raw_paths[2])
        all_smi = pd.concat([dtr, dva, dte])['compound_iso_smiles'].unique()

        graphs = {}
        for smi in tqdm(all_smi, desc="Build graphs"):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                graphs[smi] = mol_to_graph(m)
            except Exception:
                pass

        for i, p in enumerate([self.raw_paths[0],
                               self.raw_paths[1],
                               self.raw_paths[2]]):
            lst = self.process_data(p, graphs)
            data, slices = self.collate(lst)
            torch.save((data, slices), self.processed_paths[i])
            print(f"   Processed {['train', 'val', 'test'][i]}: {len(lst)}")


def create_dataloaders(train_ds, val_ds, test_ds, batch_size=16):
    """Create PyG DataLoaders for train/val/test."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
