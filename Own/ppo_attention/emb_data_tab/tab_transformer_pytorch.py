import pandas as pd
import torch
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
import torch.nn as nn


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, emb_dim,dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Cat_PreNorm(nn.Module):
    def __init__(self, emb_dim,dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, emb_dim,dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Cat_FeedForward(nn.Module):
    def __init__(self, emb_dim,dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * mult, emb_dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            emb_dim,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), attn

class Cat_Attention(nn.Module):
    def __init__(
            self,
            emb_dim,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(emb_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), attn


# transformer

class Transformer(nn.Module):
    def __init__(self, dim, emb_dim,depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(10000, emb_dim)
        self.layers = nn.ModuleList([])
        self.cat_layers = nn.ModuleList([])
        self.layer_norm = nn.LayerNorm(dim)
        self.layer_norm_cat = nn.LayerNorm(emb_dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim,dim, Attention(emb_dim,dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(emb_dim,dim, FeedForward(emb_dim,dim, dropout=ff_dropout)),
            ]))
        for _ in range(depth):
            self.cat_layers.append(nn.ModuleList([
                Cat_PreNorm(emb_dim, dim, Cat_Attention(emb_dim,dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                Cat_PreNorm(emb_dim, dim, Cat_FeedForward(emb_dim,dim, dropout=ff_dropout)),
            ]))

    def forward(self, x, return_attn=False):
        post_softmax_attns = []
        if return_attn:
            x = self.embeds(x)
            for attn, ff in self.cat_layers:
                attn_out, post_softmax_attn = attn(x)
                post_softmax_attns.append(post_softmax_attn)

                x = x + attn_out
                x = ff(x) + x
                x = self.layer_norm_cat(x)

        else:
            for attn, ff in self.layers:
                attn_out, post_softmax_attn = attn(x)
                post_softmax_attns.append(post_softmax_attn)

                x = x + attn_out
                x = ff(x) + x
                x = self.layer_norm(x)
        return x, torch.stack(post_softmax_attns)



class TabTransformer(nn.Module):
    def __init__(self,*,n_samples,emb_dim,dim,depth,heads,dim_head=16,attn_dropout=0.,ff_dropout=0.):
        super().__init__()
        self.continuous_mean_std = False
        self.transformer = Transformer(dim=dim,emb_dim=emb_dim,depth=depth,heads=heads,dim_head=dim_head,attn_dropout=attn_dropout,ff_dropout=ff_dropout)
        self.linner1 = nn.Linear(n_samples * 8, 128)
        self.norm = nn.LayerNorm(n_samples)
        self.reduce_dim = nn.Linear(n_samples,128)
        self.reduce_layer = nn.Sequential(
            # nn.BatchNorm1d(n_samples),
            nn.Linear(n_samples, dim),
            # nn.BatchNorm1d(dim),
        )
    def forward(self, x_categ, x_cont, return_attn=False):

        if x_cont is None:
            x_categ = x_categ.int()
            numpy_data = x_categ.cpu().numpy()
            cat_data = pd.DataFrame(numpy_data)
            cat_columns_names = cat_data.columns.tolist()
            categories = get_unique_categorical_counts(cat_data, cat_columns_names)
            self.num_categories = len(categories)
            self.num_unique_categories = sum(categories)
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=2)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            categories_offset = categories_offset.to('cuda')
            x_categ += categories_offset
            x_categ = x_categ.permute(1,0)
            x, attns = self.transformer(x_categ, return_attn=True)
            res_x2 = x.flatten(1)
            flat_categ = self.linner1(res_x2)
            x = flat_categ.unsqueeze(0)
            encoder_output = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
            return encoder_output
        else:
            x_cont = x_cont.permute(1,0)
            normed_cont = self.norm(x_cont)
            input = self.reduce_layer(normed_cont).unsqueeze(dim=0)

            x, attns = self.transformer(input, return_attn=False)
            encoder_output = torch.where(torch.isnan(x), torch.full_like(x, 0),x)
            return encoder_output



def get_unique_categorical_counts(dataset, cont_count):
    res_list = []
    for item in cont_count:
        if isinstance(item, str):
            res = dataset.loc[:, item].nunique()
            res_list.append(res)
        elif isinstance(item, int):
            res = dataset.iloc[:, item].nunique()
            res_list.append(res)
    return res_list