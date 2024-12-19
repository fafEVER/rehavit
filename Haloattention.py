import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


# 将设备和数据类型转换为字典格式

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

# 确保输入是元组形式
def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

# 在指定维度上扩展张量
def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 将相对位置编码转换为绝对位置编码
def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


# 生成一维的相对位置logits
def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits

# 相对位置嵌入类
class RelPosEmb(nn.Module):
    def __init__(
            self,
            block_size,
            rel_size,
            dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h


# HaloAttention类

class HaloAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            block_size,
            halo_size,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size=block_size,
            rel_size=block_size + (halo_size * 2),
            dim_head=dim_head
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
         # 验证输入特征图维度是否符合要求
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, "h or w is not divisible by block"
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'
        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=block, p2=block)

        kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c=c)

       #生成查询、键、值

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)

        # 拆分头部

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k, v))

        # 缩放查询向量

        q *= self.scale

        # 计算注意力

        sim = einsum('b i d, b j d -> b i j', q, k)

        # 添加相对位置偏置

        sim += self.rel_pos_emb(q)

        # 掩码填充

        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(mask, kernel_size=block + (halo * 2), stride=block, padding=halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b=b, h=heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # 注意力机制

        attn = sim.softmax(dim=-1)

        # 聚合

        out = einsum('b i j, b j d -> b i d', attn, v)

        # 合并和组合头部

        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
        out = self.to_out(out)

        # 将块合并回原始特征图

        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=(h // block), w=(w // block), p1=block,
                        p2=block)
        return out


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = HaloAttention(dim=512,
                          block_size=2,
                          halo_size=1, ).cuda()# 创建HaloAttention实例
    input = torch.rand(3, 512, 224, 224).cuda()# 创建随机输入
    output = block(input) # 前向传播
    print(output.shape)

