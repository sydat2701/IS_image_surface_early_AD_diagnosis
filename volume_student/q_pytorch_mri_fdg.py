import torch
import torch.nn as nn
import torch.nn.functional as F

def get_shape(tensor):
    if isinstance(tensor, torch.Tensor):
        return list(tensor.shape)

    dynamic = tensor.size()

    if tensor.shape == torch.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class PatchEmbeddings(nn.Module):
    def __init__(self, num_patches, dims, reshape=True, n_ins = 3):
        super(PatchEmbeddings, self).__init__()
        self.num_patches = num_patches
        self.proj = nn.Linear(n_ins, dims)
        self.pos_embedding = nn.Embedding(num_patches, dims)
        self.layer_norm = nn.LayerNorm(dims, eps=1e-6)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def forward(self, x):
        positions = torch.arange(0, self.num_patches).to(self.device)
        positions = self.pos_embedding(positions)

        x = torch.mean(x, dim=-1)
        
        B, C, L = x.shape
        x = x.permute(0, 2, 1).reshape(B*L, C)
        x = self.proj(x).view(B, L, -1) + positions
        x = self.layer_norm(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dims, dropout_ratio=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dims = dims
        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims)
        self.value = nn.Linear(dims, dims)
        self.proj = nn.Linear(dims, dims)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, q, x):
        B, P, C = get_shape(x)
        head_dims = self.dims // self.num_heads

        query = self.query(q).view(-1, P, head_dims, self.num_heads)
        key = self.key(x).view(-1, P, head_dims, self.num_heads)
        value = self.value(x).view(-1, P, head_dims, self.num_heads)

        heads = []
        for i in range(self.num_heads):
            q, k, v = query[:, :, :, i], key[:, :, :, i], value[:, :, :, i]
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dims ** 0.5)
            weights = F.softmax(scores, dim=-1)
            fuse_weights = weights
            weights = self.dropout(weights)
            head = torch.matmul(weights, v)
            fuse_v = v
            heads.append(head)

        if len(heads) > 1:
            x = torch.cat(heads, dim=-1)
        else:
            x = heads[0]

        x = self.proj(x)
        x = self.dropout(x)
        return x, fuse_weights, fuse_v

class MLP(nn.Module):
    def __init__(self, dims, mlp_ratio=4, dropout_ratio=0.0):
        super(MLP, self).__init__()
        # print(mlp_ratio)
        self.d1 = nn.Linear(dims, dims * mlp_ratio)
        self.bn1 = nn.BatchNorm1d(dims * mlp_ratio, eps=1e-6)
        self.d2 = nn.Linear(dims * mlp_ratio, dims)
        self.bn2 = nn.BatchNorm1d(dims, eps=1e-6)
        self.drop = nn.Dropout(dropout_ratio)

    def forward(self, x):
        '''x: (B, L, C)'''
        x = self.d1(x).transpose(1,2)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.drop(x).transpose(1,2)
        x = self.d2(x).transpose(1,2)
        x = self.bn2(x)
        x = self.drop(x)

        x = x.transpose(1,2)
        #(B, L, C)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dims, num_heads, mlp_ratio=4, dropout_ratio=0.0):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dims, eps=1e-6)
        self.ln2 = nn.LayerNorm(dims, eps=1e-6)
        self.ln_sc = nn.LayerNorm(dims, eps=1e-6)
        self.mha = MultiHeadAttention(num_heads, dims, dropout_ratio)
        self.mlp = MLP(dims, mlp_ratio, dropout_ratio)
        
    def forward(self, x, skip_connection=None):
        if skip_connection is None:
            ln_x = self.ln1(x)
            out, fuse_weight, fuse_v = self.mha(ln_x, ln_x)
            x = x + out
        else:
            out, fuse_weight, fuse_v = self.mha(self.ln1(x), self.ln_sc(skip_connection))
            x = x + out
        x = x + self.mlp(self.ln2(x))
        return x, fuse_weight, fuse_v

class AttentionModel(nn.Module):
    def __init__(self, dims, depth, heads, num_patches=20, num_classes=1, num_channels=3, num_vertices=153, dropout=0.1, branches=[], activation='sigmoid', device = 'cuda'):
        super(AttentionModel, self).__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.dims = dims
        self.heads = heads
        self.num_patches = num_patches
        self.num_vertices = num_vertices
        self.activation = activation
        self.dropout = dropout
        self.branches = branches
        self.norm = nn.LayerNorm(dims)

        self.patch_embeddx = PatchEmbeddings(self.num_patches, self.dims, n_ins=3).to(device)
        self.patch_embeddy = PatchEmbeddings(self.num_patches, self.dims, n_ins=1).to(device)
        #self.trans_att_block = TransformerBlock(self.dims, self.heads, mlp_ratio=4, dropout_ratio=self.dropout).to(device)
        # self.trans_attx = nn.ModuleList([TransformerBlock(self.dims, self.heads, mlp_ratio=4, dropout_ratio=self.dropout).to(device) for i in range(depth[0])])
        # self.trans_atty = nn.ModuleList([TransformerBlock(self.dims, self.heads, mlp_ratio=4, dropout_ratio=self.dropout).to(device) for i in range(depth[0])])
        self.trans_att = nn.ModuleList([TransformerBlock(self.dims, self.heads, mlp_ratio=4, dropout_ratio=self.dropout).to(device) for i in range(depth[0])])

        self.trans_cross_att_block1 = nn.ModuleList([TransformerBlock(self.dims, self.heads, mlp_ratio=4, dropout_ratio=self.dropout).to(device) for i in range(depth[1])])

        self.trans_cross_att_block2 = nn.ModuleList([TransformerBlock(self.dims, self.heads, mlp_ratio=4, dropout_ratio=self.dropout).to(device) for i in range(depth[1])])
        self.fc1 = nn.Linear(640*192*2, dims)
        self.norm1 = nn.LayerNorm(dims,eps=1e-6)
        self.fc2 = nn.Linear(dims, self.num_classes)
        self.sig = nn.Sigmoid()

    def reshape(self, x):
        B, P, V, C = get_shape(x)
        x = x.view(B, P, V * C)
        return x

    def forward(self, x):
        # x = x.type(torch.FloatTensor)
        # x = x.cuda() if torch.cuda.is_available() else x
        # print(">>>>>>>>>>>>>>>>>>")
        # print(x.shape)
        #x = nn.GaussianNoise(1)(x)
        x = x + (1**0.5)*torch.randn_like(x)
        x = [x[:, :3, :], x[: , 3, :].unsqueeze(1)]
        # print(x[0].shape, x[1].shape)

        x = [self.patch_embeddx(x[0]), self.patch_embeddy(x[1])]
        
        for i in range(self.depth[0]):
            # x = [self.trans_att_block(x[0]), self.trans_att_block(x[1])]
            outx, fused_attx, fuse_v_x = self.trans_att[i](x[0])
            outy, fused_atty, fuse_v_y = self.trans_att[i](x[1])
            x = [outx, outy]
            
        if len(self.depth) > 1:
            for i in range(self.depth[1]):
                x = [self.trans_cross_att_block1[i](x[0], x[1])[0],
                     self.trans_cross_att_block2[i](x[1], x[0])[0]]
        # print("*************************")
        # print("///////////// ", x[0].shape, x[0].shape)
        
        
        x = [self.norm(x[0]), self.norm(x[1])]
        x = [nn.Flatten()(y) for y in x]
        # print("///////////// ", x[0].shape, x[0].shape)

        if len(x) > 1:
            x = torch.cat(x, dim=-1)
        else:
            x = x[0]

        x = self.fc1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.fc2(x)
        x = self.sig(x)
        
        return x, fused_attx, fuse_v_x, fused_atty, fuse_v_y
if __name__ == '__main__':
    model = AttentionModel(dims=192,
                               depth=[3,3],
                               heads=3,
                               num_patches=640,
                               num_classes=1,
                               num_channels=4,
                               num_vertices=153,
                               dropout=0.1,
                               branches=[slice(0, 3), slice(3, 4)],
                               activation='sigmoid').cuda()
    x = torch.rand((2, 4, 640, 153)).float().cuda()
    out = model(x)
    print(out.shape)