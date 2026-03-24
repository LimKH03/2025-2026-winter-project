from torch import nn

class GEGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.half_dmodel = d_model//2
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = x[... , :self.half_dmodel]
        x2  = x[..., self.half_dmodel: ]
        x1 = self.gelu(x1)
        x= x1*x2
        return x

class AttentionLayer(nn.Module): #interaction layer
    def __init__(self, d_model, d_ff, n_heads, drop_out, loops = 1):
        super().__init__()

        self.loop_count = loops
        self.dropout = nn.Dropout(drop_out)

        self.atten_LN = nn.ModuleList(
            [ 
              nn.LayerNorm(d_model) for _ in range(loops)
            ]
        )
        self.cross_atten = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, n_heads,  dropout=drop_out, batch_first=True) 
                for _ in range(loops)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.FF = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model,d_model*4),
                    GEGLU(d_model*4),
                    nn.Linear(d_model*2,d_model)
                ) for _ in range(loops)
            ]
        )
    def forward(self,hyp, src, enc_mask=None):

        for LN, attn, ff in zip(self.atten_LN, self.cross_atten, self.FF):
            x = hyp

            hyp = LN(x)
            hyp = attn(hyp,src,src)[0]
            hyp = hyp + x

            x= hyp
            hyp = ff(hyp)
            hyp = hyp+x
            

        return self.final_norm(hyp)