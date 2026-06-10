from typing import Optional
from jittor import nn

import jittor as jt

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__()
        
        if activation == 'ReLU':
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU()
            )
            self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise Exception("Please assign valid activation to MLP!")
    
    def execute(self, x, edge_index):
        """
        x: (N, C)
        edge_index: (2, E)
        """
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        
        # gather
        x_i = x[dst]  # (E, C)
        x_j = x[src]  # (E, C)
        
        # message
        tmp = jt.concat([x_i, x_j - x_i], dim=1)  # (E, 2C)
        msg = self.mlp(tmp)  # (E, out_channels)
        
        N = x.shape[0]
        C = msg.shape[1]

        # E = N * k
        k = msg.shape[0] // N

        # (N, k, C)
        msg = msg.reshape(N, k, C)

        # max pooling over neighbors
        out = jt.max(msg, dim=1)
        # out = jt.full((N, msg.shape[1]), 0)
        # cnt = jt.full((N, msg.shape[1]), 0)
        
        # # scatter mean
        # out = out.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), msg, reduce='add')
        # cnt = cnt.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), jt.ones_like(msg), reduce='add')
        # out = out / (cnt + 1e-8)
        out_2 = self.lin(x)
        return out + out_2

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__(in_channels, out_channels, activation)
    
    def execute(self, x, edge_index):
        return super().execute(x, edge_index)

class FeatureExtraction(nn.Module):
    def __init__(self, k=32, input_dim=0, embedding_dim=512, distance_estimation=False):
        super().__init__()

        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.distance_estimation = distance_estimation

        self.conv1 = DynamicEdgeConv(self.input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            embedding_dim,
            activation=None
        )
    # ========= edge_index 构建 =========
    def get_edge_index(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape
        knn_idx = get_knn_idx(x, x, self.k + 1)  # (B, N, k+1)
        knn_idx = knn_idx[:, :, 1:]
        base = jt.arange(B) * N  # (B,)
        base = base.reshape(B, 1, 1)
        
        knn_idx = knn_idx + base  # (B, N, k)
        
        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, self.k))
        dst = dst + base
        
        src = knn_idx.reshape(-1)
        dst = dst.reshape(-1)
        
        edge_index = jt.stack([src, dst], dim=0)  # (2, E)
        
        return edge_index
    
    def normalize_patch(self, pcl):
        scale = jt.sqrt((pcl ** 2).sum(-1, keepdims=True))
        scale = scale.max(dim=-2, keepdims=True)
        return pcl / (scale + 1e-8) # type: ignore
    
    def execute(self, x):
        # x: (B, N, C)
        B, N, _ = x.shape
        
        if self.distance_estimation:
            x = self.normalize_patch(x)
        
        # -------- conv1 --------
        edge_index = self.get_edge_index(x)
        x_flat = x.reshape(B * N, -1)
        
        x1 = self.conv1(x_flat, edge_index)
        x1 = x1.reshape(B, N, -1)
        
        # -------- conv2 --------
        edge_index = self.get_edge_index(x1)
        x1_flat = x1.reshape(B * N, -1)
        
        x2 = self.conv2(x1_flat, edge_index)
        x2 = x2.reshape(B, N, -1)
        
        # -------- conv3 --------
        edge_index = self.get_edge_index(x2)
        
        x_combined = jt.concat([x1, x2], dim=-1)
        x_combined_flat = x_combined.reshape(B * N, -1) # type: ignore
        
        x3 = self.conv3(x_combined_flat, edge_index)
        x3 = x3.reshape(B, N, -1)
        
        return x3

class Decoder(nn.Module):
    
    def __init__(self, z_dim, dim, out_dim, hidden_size):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        c_dim = z_dim
        self.lin_1 = nn.Linear(c_dim, c_dim)
        self.bn_1_out = nn.BatchNorm1d(c_dim)
        
        self.lin_2 = nn.Linear(c_dim, hidden_size)
        self.bn_2_out = nn.BatchNorm1d(hidden_size)
        
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        
        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def execute(self, c, B=None, N=None):
        """
        c: (B*N, F)
        """
        net = self.lin_1(c)
        net = self.bn_1_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        net = self.lin_2(net)
        net = self.bn_2_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        if self.out_dim == 1:
            net = net.reshape(B, N, -1)
            net = jt.max(net, dim=1, keepdims=True)
            net = self.lin_3(net)
            net = jt.sigmoid(net)
        else:
            net = self.lin_3(net)
        return net

def get_knn_idx(x, y, k, offset=0):
    """
    x: (B, N, d)
    y: (B, M, d)
    return: (B, N, k)
    """
    K = k + offset
    # Use distance-based KNN for better stability with large k values
    # Instead of jt.misc.knn which may have issues with k > 32
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    _, idx = jt.topk(dist, k=K, dim=-1, largest=False)
    return idx[:, :, offset:]


# ==================== Enhanced Components ====================

class MultiScaleEdgeConv(nn.Module):
    """Multi-scale EdgeConv for capturing features at different neighbor scales."""
    def __init__(self, in_channels, out_channels, k_list=[8, 16, 32], activation='ReLU'):
        super().__init__()
        self.k_list = k_list
        self.num_scales = len(k_list)
        self.out_channels = out_channels

        # Each conv outputs full out_channels (better per-scale representation)
        self.convs = nn.ModuleList([
            EdgeConv(in_channels, out_channels, activation)
            for _ in range(self.num_scales)
        ])

        # Fusion: concat num_scales * out_channels → out_channels
        fused_channels = self.num_scales * out_channels
        self.fusion = nn.Sequential(
            nn.Linear(fused_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def execute(self, x, edge_indices_dict):
        """
        x: (N, C)
        edge_indices_dict: dict with k values as keys, each edge_index is (2, E)
        """
        multi_scale_features = []

        for k, conv in zip(self.k_list, self.convs):
            edge_index = edge_indices_dict[k]
            feat = conv(x, edge_index)
            multi_scale_features.append(feat)

        # Concatenate multi-scale features (each is out_channels)
        out = jt.concat(multi_scale_features, dim=-1)  # (N, num_scales * out_channels)
        out = self.fusion(out)  # (N, out_channels)

        return out


class EnhancedFeatureExtractor(nn.Module):
    """
    Simplified multi-scale feature extraction (like original VM but with multi-scale).
    - Uses MultiScaleEdgeConv instead of single-scale EdgeConv
    - No residual blocks (kept simple like original VM)
    """
    def __init__(self, k_list=[8, 16, 32], input_dim=3, embedding_dim=256):
        super().__init__()

        self.k_list = k_list
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Multi-scale EdgeConv layers (same structure as original FeatureExtraction)
        # conv1: directly from input_dim (like original VM)
        self.conv1 = MultiScaleEdgeConv(
            self.input_dim,
            embedding_dim // 8,
            k_list=k_list
        )

        self.conv2 = MultiScaleEdgeConv(
            embedding_dim // 8,
            embedding_dim // 4,
            k_list=k_list
        )

        self.conv3 = MultiScaleEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            embedding_dim,
            k_list=k_list,
            activation=None
        )

    def get_edge_indices_dict(self, x):
        """
        x: (B, N, C)
        return: dict of edge_index for each k value
        """
        B, N, _ = x.shape
        edge_indices = {}

        for k in self.k_list:
            knn_idx = get_knn_idx(x, x, k + 1)  # (B, N, k+1)
            knn_idx = knn_idx[:, :, 1:]  # Remove self

            base = jt.arange(B) * N  # (B,)
            base = base.reshape(B, 1, 1)

            knn_idx = knn_idx + base  # (B, N, k)

            # Use arange and repeat instead of broadcast
            dst = jt.arange(N).reshape(1, N, 1)
            dst = jt.repeat(dst, (B, 1, k))
            dst = dst + base

            src = knn_idx.reshape(-1)
            dst = dst.reshape(-1)

            edge_index = jt.stack([src, dst], dim=0)  # (2, E)
            edge_indices[k] = edge_index

        return edge_indices

    def execute(self, x):
        """
        x: (B, N, 3) - input point cloud coordinates
        return: (B, N, embedding_dim) - feature embeddings
        """
        B, N, _ = x.shape

        # -------- conv1 -------- (directly from input x, like original VM)
        edge_indices = self.get_edge_indices_dict(x)
        x_flat = x.reshape(B * N, -1)

        x1 = self.conv1(x_flat, edge_indices)
        x1 = x1.reshape(B, N, -1)

        # -------- conv2 --------
        edge_indices = self.get_edge_indices_dict(x1)
        x1_flat = x1.reshape(B * N, -1)

        x2 = self.conv2(x1_flat, edge_indices)
        x2 = x2.reshape(B, N, -1)

        # -------- conv3 --------
        edge_indices = self.get_edge_indices_dict(x2)

        x_combined = jt.concat([x1, x2], dim=-1)
        x_combined_flat = x_combined.reshape(B * N, -1)

        x3 = self.conv3(x_combined_flat, edge_indices)
        x3 = x3.reshape(B, N, -1)

        return x3
