import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch_geometric.nn import global_mean_pool, GraphConv


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class NodeLevelBatchNorm(_BatchNorm):
    """BatchNorm1d variant for node-level features in graph batches."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, x):
        if x.dim() != 2:
            raise ValueError('need 2D')

    def forward(self, x):
        self._check_input_dim(x)
        eaf = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                eaf = 1.0 / float(self.num_batches_tracked)
        return F.batch_norm(x, self.running_mean, self.running_var,
                            self.weight, self.bias,
                            self.training or not self.track_running_stats,
                            eaf, self.eps)


# ---------------------------------------------------------------------------
# GNN building blocks
# ---------------------------------------------------------------------------
class GraphConvBn(nn.Module):
    """GraphConv + BatchNorm + ReLU."""

    def __init__(self, i, o):
        super().__init__()
        self.conv = GraphConv(i, o)
        self.norm = NodeLevelBatchNorm(o)
        self.act = nn.ReLU()

    def forward(self, x, ei):
        return self.act(self.norm(self.conv(x, ei)))


class DenseLayer(nn.Module):
    """Two GraphConv+BN+ReLU layers (GNN layers = 2)."""

    def __init__(self, nin, growth=32, bn=4):
        super().__init__()
        self.c1 = GraphConvBn(nin, growth * bn)
        self.c2 = GraphConvBn(growth * bn, growth)

    def forward(self, x, ei):
        return self.c2(self.c1(x, ei), ei)


class DenseBlock(nn.Module):
    """Dense block: each layer receives concatenation of all previous features."""

    def __init__(self, L, nin, growth=32, bn=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [DenseLayer(nin + i * growth, growth, bn) for i in range(L)]
        )

    def forward(self, x, ei):
        feats = [x]
        for l in self.layers:
            feats.append(l(torch.cat(feats, 1), ei))
        return torch.cat(feats, 1)


class TransitionLayer(nn.Module):
    """Transition layer: GraphConv+BN+ReLU for dimensionality reduction."""

    def __init__(self, i, o):
        super().__init__()
        self.conv = GraphConvBn(i, o)

    def forward(self, x, ei):
        return self.conv(x, ei)


# ---------------------------------------------------------------------------
# MGNN: DenseNet-style GNN encoder
# ---------------------------------------------------------------------------
class MGNN(nn.Module):
    """
    Multi-layer GNN encoder with DenseNet-style blocks.

    Args:
        num_input_features: input atom feature dimension (default 87)
        out_dim: output embedding dimension
        growth_rate: growth rate for dense blocks
        block_config: number of layers in each dense block
        bn_sizes: bottleneck multiplier per block
    """

    def __init__(self, num_input_features=87, out_dim=320, growth_rate=32,
                 block_config=(8, 8, 8), bn_sizes=(2, 2, 2)):
        super().__init__()
        self.initial_conv = GraphConvBn(num_input_features, 32)
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        nf = 32
        for i, (L, bs) in enumerate(zip(block_config, bn_sizes)):
            self.blocks.append(DenseBlock(L, nf, growth_rate, bs))
            nf += L * growth_rate
            if i < len(block_config) - 1:
                self.transitions.append(TransitionLayer(nf, nf // 2))
                nf //= 2
        self.projection = nn.Linear(nf, out_dim)
        self.out_dim = out_dim

    def forward(self, x, ei, batch):
        """
        Returns:
            node_feats: per-node features (N, out_dim)
            graph_emb: graph-level embedding (B, out_dim)
        """
        x = self.initial_conv(x, ei)
        for i, b in enumerate(self.blocks):
            x = b(x, ei)
            if i < len(self.transitions):
                x = self.transitions[i](x, ei)
        return x, self.projection(global_mean_pool(x, batch))


# ---------------------------------------------------------------------------
# Bidirectional cross-attention
# ---------------------------------------------------------------------------
class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between compound node features and protein
    embeddings.

    Two directions:
        - Compound-to-Protein (c2p): each atom queries protein residues
        - Protein-to-Compound (p2c): each residue queries compound atoms
    """

    def __init__(self, embed_dim=320, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Compound-to-Protein projections
        self.W_Q_C = nn.Linear(embed_dim, embed_dim)
        self.W_K_P = nn.Linear(embed_dim, embed_dim)
        self.W_V_P = nn.Linear(embed_dim, embed_dim)

        # Protein-to-Compound projections
        self.W_Q_P = nn.Linear(embed_dim, embed_dim)
        self.W_K_C = nn.Linear(embed_dim, embed_dim)
        self.W_V_C = nn.Linear(embed_dim, embed_dim)

        # Output projections
        self.out_proj_C = nn.Linear(embed_dim, embed_dim)
        self.out_proj_P = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _c2p(self, compound, protein):
        """Compound-to-Protein attention."""
        B, N, _ = compound.shape
        _, R, _ = protein.shape
        Q = self.W_Q_C(compound).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K_P(protein).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V_P(protein).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        w = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        w = self.dropout(w)
        o = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj_C(o) + compound

    def _p2c(self, compound, protein):
        """Protein-to-Compound attention."""
        B, R, _ = protein.shape
        _, N, _ = compound.shape
        Q = self.W_Q_P(protein).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K_C(compound).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V_C(compound).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        w = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        w = self.dropout(w)
        o = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, R, self.embed_dim)
        return self.out_proj_P(o) + protein

    def forward(self, compound, protein, use_c2p=True, use_p2c=True):
        """
        Args:
            compound: (B, N, D) atom-level features
            protein: (B, R, D) residue-level features
            use_c2p: enable compound-to-protein attention
            use_p2c: enable protein-to-compound attention

        Returns:
            uc: updated compound representation
            up: updated protein representation
        """
        uc, up = compound, protein
        if use_c2p:
            uc = self._c2p(compound, protein)
        if use_p2c:
            up = self._p2c(compound, protein)
        return uc, up


# ---------------------------------------------------------------------------
# Full BiCrossAtt-CPI model
# ---------------------------------------------------------------------------
class BiCrossAttCPI(nn.Module):
    """
    BiCrossAtt-CPI: Bidirectional Cross-Attention Deep Hybrid model for CPI.

    Ablation flags:
        use_compound_gnn    : if False, replace MGNN with a simple MLP on raw atom features
        use_cross_attention : if False, skip cross-attention entirely
        use_c2p_attention   : if False, disable compound->protein attention
        use_p2c_attention   : if False, disable protein->compound attention
    """

    def __init__(self,
                 compound_input_dim=87,
                 protein_embed_dim=640,
                 cross_attn_heads=8,
                 cross_attn_dropout=0.1,
                 num_classes=2,
                 use_compound_gnn=True,
                 use_cross_attention=True,
                 use_c2p_attention=True,
                 use_p2c_attention=True,
                 dropout_rate=0.2):
        super().__init__()
        self.use_compound_gnn = use_compound_gnn
        self.use_cross_attention = use_cross_attention
        self.use_c2p_attention = use_c2p_attention
        self.use_p2c_attention = use_p2c_attention
        self.protein_embed_dim = protein_embed_dim

        if use_compound_gnn:
            self.mgnn = MGNN(num_input_features=compound_input_dim,
                             out_dim=protein_embed_dim,
                             growth_rate=32, block_config=[8, 8, 8])
            self.project_compound_nodes = nn.Linear(456, protein_embed_dim)
        else:
            self.simple_compound_proj = nn.Sequential(
                nn.Linear(compound_input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, protein_embed_dim)
            )

        if use_cross_attention:
            self.cross_attention = BidirectionalCrossAttention(
                embed_dim=protein_embed_dim,
                num_heads=cross_attn_heads,
                dropout=cross_attn_dropout)

        self.classifier = nn.Sequential(
            nn.Linear(protein_embed_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, compound_x, compound_edge_index, compound_batch,
                protein_embeddings):
        B = protein_embeddings.shape[0]

        # --- Compound encoding ---
        if self.use_compound_gnn:
            node_feats, _ = self.mgnn(compound_x, compound_edge_index, compound_batch)
            node_feats = self.project_compound_nodes(node_feats)
        else:
            node_feats = self.simple_compound_proj(compound_x)

        # Pad node features to (B, max_atoms, D)
        node_list = [node_feats[compound_batch == i] for i in range(B)]
        max_atoms = max(1, max(len(n) for n in node_list)) if node_list else 1
        padded = torch.zeros(B, max_atoms, node_feats.shape[-1
