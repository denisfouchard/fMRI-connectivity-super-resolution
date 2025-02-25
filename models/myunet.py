class MYGraphUnet(nn.Module):

    def __init__(self, ks, dim, act, drop_p, in_dim, out_dim):
        super(MYGraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.n_layers = len(ks)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.upscale_proj = nn.Linear(in_features=in_dim, out_features=out_dim)

        # First down block
        self.down_blocks.append(GCN(dim, dim))
        self.pools.append(KPool(ks[0], dim, drop_p))

        # Middle down/up-blocks
        for i in range(1, self.n_layers - 1):
            self.down_blocks.append(GCN(dim, dim))
            self.up_blocks.append(GCN(dim, dim))
            self.pools.append(KPool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

        self.up_blocks.append(Unpool(dim, dim, drop_p))
        self.up_blocks.append(GCN(dim, dim, False))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, A, n_feat):
        A = A.to(self.device)
        adj_ms = []
        indices_list = []
        down_outs = []

        X_ = torch.ones(A.shape[0], A.shape[1], n_feat, device=self.device)
        for i in range(len(self.down_blocks)):
            X = self.down_blocks[i](A, X_)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gcn(A, X)
        for i in range(len(self.down_blocks)):
            up_idx = self.n_layers - i - 1
            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A = nn.functional.sigmoid(X @ X.T)
            A, X = self.unpools[i](A, X, down_outs[up_idx], idx)
            X = self.up_blocks[i](A, X)
            X = X + (down_outs[up_idx])  # Skip connection

        X = self.upscale_proj(X)
        A = nn.functional.sigmoid(X @ X.T)
        return A


class GCN(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN).
    """

    def __init__(self, input_dim, output_dim, use_nonlinearity=True):
        super(GCN, self).__init__()
        self.use_nonlinearity = use_nonlinearity
        self.Omega = nn.Parameter(
            torch.randn(input_dim, output_dim)
            * torch.sqrt(torch.tensor(2.0) / (input_dim + output_dim))
        )
        self.beta = nn.Parameter(torch.zeros(output_dim))

    def forward(self, A_normalized, H_k):
        agg = torch.matmul(A_normalized, H_k)  # local agg
        H_k_next = torch.matmul(agg, self.Omega) + self.beta
        return nn.functional.relu(H_k_next) if self.use_nonlinearity else H_k_next


class KPool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(KPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()  # added dropout here

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, A, X, pre_h, idx):
        new_h = X.new_zeros([A.shape[0], X.shape[1]])
        new_h[idx] = X
        return A, new_h


def top_k_graph(scores, A, X, k):
    print(A.shape)
    num_nodes = A.shape[1]
    values, idx = torch.topk(
        scores, max(2, int(k * num_nodes))
    )  # make sure k works based on number of current nodes
    print("X shape:", X.shape)
    print("idx shape : ", idx.shape)
    print(idx)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, X.shape[-1])
    new_h = torch.gather(X, 1, idx_expanded)
    values = torch.unsqueeze(values, -1)
    print("H shape", new_h.shape)
    new_h = torch.mul(new_h, values)
    un_g = A.bool().float()
    un_g = (
        torch.matmul(un_g, un_g).bool().float()
    )  # second power to reduce chance of isolated nodes
    un_g = torch.gather(
        un_g, 1, idx.unsqueeze(2).expand(-1, -1, A.shape[-1])
    )  # Shape: (32, 56, 112)

    # Gather along width dimension (dim=2)
    un_g = torch.gather(un_g, 2, idx.unsqueeze(1).expand(-1, idx.shape[1], -1))
    print("un_g shape", un_g.shape)
    # A = norm_g(un_g)
    return A, new_h, idx
