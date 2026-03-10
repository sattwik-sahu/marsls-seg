import torch
import torch.nn as nn 
import torch.nn.functional as F


class Tokens2FPN(nn.Module):
    
    """
        DESCR: 
                -This class converts the tokens to the feature pyramid network that is used by the FPN Module
                - Reshapes the token back to spatial and builds a four scale pyramid out of that.
        TODO: Defines the attribute of the class
    """
    def  __init__(self,embed_dim:int,fpn_dim:int=256)->None:
        super().__init__()
        self._embed_dim:int=embed_dim
        self._fpn_dim:int=fpn_dim
        self._layer_norm=nn.LayerNorm(self._embed_dim)
        self._projection=nn.Sequential(
            nn.Conv2d(in_channels=self._embed_dim,out_channels=self._fpn_dim,kernel_size=1,bias=False),
            nn.GroupNorm(32,self._fpn_dim),
            nn.GELU()
        )
        self._to_res3=self._up(self._fpn_dim)
        self._to_res2=self._up(self._fpn_dim)
        self._to_res5=self._down(self._fpn_dim)




    @staticmethod
    def _up(c:int):
        return nn.Sequential(
            nn.ConvTrasnpose2d(in_chanels=c,out_channles=c,kernel_size=2,stride=2,bias=False),
            nn.GroupNorm(32,c),
            nn.GELU(),
            nn.Conv2d(in_channels=c,out_channels=c,kernel_size=3,padding=1,bias=False),
            nn.GroupNorm(32,c),
            nn.GELU())
    
    @staticmethod                      
    def _down(c:int):
        return nn.Sequential(
            nn.Conv2d(in_channels=c,out_channels=c,kernel_size=3,stride=2,padding=1,bias=False),
            nn.GroupNorm(32,c),
            nn.GELU()

        )
    def forward(self,x:torch.Tensor,h:int,w:int)->dict:
        B,N,C=x.shape

        x=x.permute(0,2,1)
        x=x.view(B,C,h,w)
        res4=self._projection(x)
        res3=self._to_res3(res4)
        res2=self._to_res2(res3)
        res5=self._to_res5(res4)


        return {
            "res2":res2,
            "res3":res3,
            "res4":res4,
            "res5":res5
        }  



class PixelDecoder(nn.Module):
    """
        DESCR: This class fuses the different layers of the pyramid
        TODO: Define the attribute of the class
    """ 
    def __init__(self,fpn_dim:int=256,mask_dim=256):
        super().__init__()
        self._fpn_dim:int=fpn_dim
        self._mask_dim:int=mask_dim
        self._lat5=nn.Conv2d(self._fpn_dim,self._fpn_dim,1)
        self._lat4=nn.Conv2d(self._fpn_dim,self._fpn_dim,1)
        self._lat3=nn.Conv2d(self._fpn_dim,self._fpn_dim,1)
        self._lat2=nn.Conv2d(self._fpn_dim,self._fpn_dim,1)
        self._out5=self._block(self._fpn_dim)
        self._out4=self._block(self._fpn_dim)
        self._out3=self._block(self._fpn_dim)

        self._projection=nn.Conv2d(in_channels=self._fpn_dim,out_channels=self._mask_dim,kernel_size=3,padding=1)
    @staticmethod
    def _block(c:int):
        return nn.Sequential(
            nn.Conv2d(in_channels=c,out_channels=c,kernel_size=3,padding=1),
            nn.GroupNorm(32,c),
            nn.GELU()
        )
    

    def forward(self,feats:dict):
        p5=self._out5(self._lat5(feats['res5']))
        p4=self._out4(self._lat4(feats['res4']))+ F.interpolate(p5,scale_factor=2,mode="nearest")
        p3=self._out3(self._lat3(feats['res3'])) + F.interpolate(p4,scale_factor=2,mode="nearest")
        p2=self._lat2(feats['res5'])+ F.interpolate(p3,scale_factor=2,mode="nearest")
        mask_features=self._projection(p2)

        encoder_features=[p5,p4,p3]
        return mask_features,encoder_features
    

class DecoderLayer(nn.Module):
    def __init__(self, dim:int=256,heads:int=8,ffn_dim:int=2048):
        super().__init__()
        self._dim:int=dim
        self._heads:int=heads
        self._ffn_dim:int=ffn_dim

        self._cross_attention=nn.MultiheadAttention(self._dim,self._heads,batch_first=True)
        self._self_attention=nn.MultiheadAttention(self._dim,self._heads,batch_first=True)

        self._ffn=nn.Sequential(
            nn.Linear(self._dim,self._ffn_dim),
            nn.GELU(),
            nn.Linear(self._ffn_dim,self._dim)
        )

        self._n1=nn.LayerNorm(self._dim)
        self._n2=nn.LayerNorm(self._dim)
        self._n3=nn.LayerNorm(self._dim)


    def forward(self,q,mem,attention_mask=None):
        ca,_=self._cross_attention(q,mem,mem,attention_mask)
        q=self._n1(q+ca)
        sa=self._self_attention(q,q,q)
        q=self._n2(q+sa)

        return self._n3(q+self._ffn(q))
    


class TransformerDecoder(nn.Module):
    def __init__(self,num_queries:int=100,heads:int=8,num_classes:int=2,num_layers:int=9,mask_dim:int=128,dim:int=256,ffn_dim:int=2048)->None:
       super().__init__()
       self._num_queries:int=num_queries
       self._heads:int=heads
       self._num_classes:int=num_classes
       self._num_layers:int=num_layers
       self._mask_dim:int=mask_dim
       self._dim:int=dim
       self._ffn_dim:int=ffn_dim

       self._query_feat=nn.Embedding(self._num_queries,self._dim)
       self._query_pos=nn.Embedding(self._num_queries,self._dim)


       self._mem_proj=nn.Linear(self._dim,self._dim)

       self._layers=nn.ModuleList([
           DecoderLayer(dim=self._dim,heads=self._heads,ffn_dim=self._ffn_dim)
           for _ in range(self._num_layers)
       ])
       self._mask_heads=nn.ModuleList([
           nn.Sequential(
               nn.Linear(self._dim,self._dim),
               nn.GELU(),
               nn.Linear(self._dim,self._mask_dim)

           )
           for _ in range(self._num_layers)
       ])

       self._class_heads=nn.ModuleList([
           nn.Linear(self._dim,self._num_classes)
           for _ in range(self._num_layers)
       ])

       self._norm1=nn.LayerNorm(dim)

    def _build_attn_mask(self, mask_logits, h, w, B, nheads):
        m = F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False)
        m = (m.sigmoid() < 0.5).flatten(2)                     
        m = m | (m.sum(-1, keepdim=True) == h * w)             
        return m.unsqueeze(1).expand(-1, nheads, -1, -1).reshape(B * nheads, m.shape[1], h * w)

    def forward(self, mask_features, memory):
        B = mask_features.shape[0]
        q = self._query_feat.weight[None].expand(B, -1, -1)
        q = q + self._query_pos.weight[None].expand(B, -1, -1)

        prev_mask = None
        for i, layer in enumerate(self._layers):
            mem      = memory[i % len(memory)]
            h, w     = mem.shape[2], mem.shape[3]
            flat     = self._mem_proj(mem.flatten(2).transpose(1, 2))   

            attn_mask = None
            if prev_mask is not None:
                attn_mask = self._build_attn_mask(prev_mask, h, w, B, layer.nheads)

            q          = layer(q, flat, attn_mask)
            q_n        = self._norm(q)
            mask_embed = self._mask_heads[i](q_n)                       
            prev_mask  = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        class_logits = self._class_heads[-1](self.norm(q))              
        return prev_mask, class_logits
    





class LandslideSegHead(nn.Module):
    """
    Takes fused JEPA tokens → returns final pixel-level segmentation map.

    Args:
        jepa_dim    : dim of fused JEPA tokens         (default 768)
        fpn_dim     : internal channel dim of FPN       (default 256)
        num_queries : number of mask candidates         (default 100)
        num_classes : foreground classes excl. no-obj  (default 2)
        num_layers  : transformer decoder depth         (default 9)
    """
    def __init__(self,
                 jepa_dim:    int = 768,
                 fpn_dim:     int = 256,
                 num_queries: int = 100,
                 num_classes: int = 2,
                 num_layers:  int = 9):
        super().__init__()

        
        self._adapter     = Tokens2FPN(embed_dim=jepa_dim, fpn_dim=fpn_dim)
        self._pixel_dec   = PixelDecoder(fpn_dim=fpn_dim, mask_dim=fpn_dim)
        self._transformer = TransformerDecoder(
            num_queries = num_queries,
            dim         = fpn_dim,
            num_layers  = num_layers,
            mask_dim    = fpn_dim,
            num_classes = num_classes,
        )

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            tokens : (B, N, jepa_dim)  — fused JEPA token stream
            h, w   : token grid size   — (H_img // 16, W_img // 16)

        Returns:
            seg_map : (B, H_img, W_img) — integer class label per pixel
        """
        
        fpn = self._adapter(tokens, h, w)

       
        mask_feat, memory = self._pixel_dec(fpn)

        
        masks, class_logits = self._transformer(mask_feat, memory)

      
        H_img, W_img = h * 16, w * 16
        masks = F.interpolate(
            masks, size=(H_img, W_img), mode="bilinear", align_corners=False
        ).sigmoid()                                                   

       
        scores   = class_logits.softmax(-1)[..., :-1].max(-1).values  
        weighted = masks * scores[:, :, None, None]                    

        
        best_query = weighted.argmax(1)                                
        best_class = class_logits.softmax(-1)[..., :-1].argmax(-1)    
        seg_map    = best_class.gather(
            1, best_query.flatten(1)
        ).reshape_as(best_query)                                       

        return seg_map
