from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math

from mmcv.runner import auto_fp16, force_fp32
from timm.models.layers import trunc_normal_
from mmseg.models.losses import accuracy

from einops import rearrange


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        

def Sinkhorn_log_exp_sum(C, mu, nu, epsilon):
    
    def _log_boltzmann_kernel(u, v, epsilon, C=None):
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= epsilon
        return kernel
  
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    thresh = 1e-6
    max_iter = 100
            
    for i in range(max_iter):
       
        u0 = u  # useful to check the update
        K = _log_boltzmann_kernel(u, v, epsilon, C)
        u_ = torch.log(mu + 1e-8) - torch.logsumexp(K, dim=2)
        u = epsilon * u_ + u
        
        K_t = _log_boltzmann_kernel(u, v, epsilon, C).permute(0, 2, 1).contiguous()
        v_ = torch.log(nu + 1e-8) - torch.logsumexp(K_t, dim=2)
        v = epsilon * v_ + v
        
        err = (u - u0).abs().mean()
        if err.item() < thresh:
            break
    
    K = _log_boltzmann_kernel(u, v, epsilon, C)
    T = torch.exp(K)

    return T


class AttentionOT(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.eps = 0.05 

    def forward(self, xq, xk, xv):
        Nq, K, B, C = xq.size() 
        Nv = xv.size()[1]

        xq = self.q(xq)
        xk = self.k(xk)
        v = self.v(xv)
        
        # assign variables
        _, M, _ = xk.shape
        xq = F.normalize(xq, dim=-1, p=2)
        xk = F.normalize(xk, dim=-1, p=2)

        # compute score map 
        sim = torch.einsum('bmc,nkbc->bnkm', xk, xq)
        sim = sim.permute(0,2,3,1) 
        sim = sim.contiguous().view(B*K, M, Nq) 
        wdist = 1.0 - sim

        # optimally transport score map
        xx = torch.zeros(B*K, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy = torch.zeros(B*K, Nq, dtype=sim.dtype, device=sim.device).fill_(1. / Nq)
        T = Sinkhorn_log_exp_sum(wdist, xx,yy, self.eps)
        
        # T * score map
        score_map = (M * Nq * sim * T).view(B, K, M, Nq) 
        attn_save = score_map.clone().contiguous().sum(dim=-1).squeeze(-1)
        attn = rearrange(T.view(B, K, M, Nq), 'b k m n -> n b k m', b = B, k = K, n = Nq) 
        attn = self.attn_drop(attn)

        x = torch.einsum('nbkm,bmc->knbc', attn, v)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn_save


class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None: 
            output = self.norm(output)

        return outputs, attns


class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn

        # MPSA (Multi-prompts Sinkhorn Attention)
        self.multihead_attn = AttentionOT( 
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2


@HEADS.register_module()
class ATMSingleHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=1,
            use_proj=True,
            crop_train=False,
            **kwargs,
    ):
        super(ATMSingleHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []

        self.unseen_idx = self.all_idx.copy()
        for i_idx in self.seen_idx:
            self.unseen_idx.remove(i_idx)

        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)

            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)

            # decoder layer
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders

        delattr(self, 'conv_seg')

        self.q_proj = nn.Linear(dim * 2, dim)
        self.eps = 0.05 

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, self_training=False, st_mask=None):
        seg_logits, score_map = self.forward(inputs)

        if self_training:
            self_balancing_factor  = 0.5
            pseudo_semantic_masks = self_balancing_factor*score_map['pred_masks'].clone().detach() +(1-self_balancing_factor)*seg_logits['pred_masks'].clone().detach()            
            pseudo_semantic_masks = pseudo_semantic_masks.sigmoid()
            
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1)

            # generate pseudo labels for "transductive" setting
            gt_semantic_seg[gt_semantic_seg==-1] = pseudo_semantic_seg[gt_semantic_seg==-1]
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        else:
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)
        
        losses_score = self.losses(score_map, gt_semantic_seg)

        return losses, losses_score

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training)

    def forward(self, inputs_both, self_training=None):
        inputs = inputs_both[0][0]
        cls_token = inputs_both[0][1]
        text_token = inputs_both[1]
        
        x = []
        for stage_ in inputs[:self.use_stages]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()

        laterals = []
        attns = []
        maps_size = []
        qs = []

        for idx, (x_, proj_, norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
            lateral = norm_(proj_(x_))
            if idx == 0:
                laterals.append(lateral)
            else:
                if laterals[idx - 1].size()[1] == lateral.size()[1]:
                    laterals.append(lateral + laterals[idx - 1])
                else:
                    # nearest interpolate
                    l_ = self.d3_to_d4(laterals[idx - 1])
                    l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
                    l_ = self.d4_to_d3(l_)
                    laterals.append(l_ + lateral)

        # Relationship Descriptor
        lateral = laterals[0]
        q = self.q_proj(self.get_qs(text_token, cls_token))  
        pred_score_map = None
        
        B, C, H, W = inputs[-1].shape  
        _, K, N, C = q.shape
        
        # (a) MPS path
        text = F.normalize(q, dim=-1, p=2)
        for visual_embeddings in laterals:
            
            visual_embeddings = F.normalize(visual_embeddings, dim=-1, p=2)
            _, M, _ = visual_embeddings.shape

            # compute score map 
            sim = torch.einsum('bmc,bknc->bnkm', visual_embeddings, text)
            sim = sim.permute(0,2,3,1)
            sim = sim.contiguous().view(B*K, M, N) 
            wdist = 1.0 - sim

            # MPS (Multi-Prompts Sinkhorn)
            xx = torch.zeros(B*K, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
            yy = torch.zeros(B*K, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)
            T = Sinkhorn_log_exp_sum(wdist, xx,yy, self.eps)
                
            score_map = (M * N * T * sim) 
            score_map = score_map.contiguous().view(B, K, H, W, N).sum(dim=-1).squeeze(-1) 
            
        score_map = score_map.contiguous().view(B, K, H, W)  
        score_map = F.interpolate(score_map, size=(self.image_size, self.image_size),mode='bilinear', align_corners=False)
        pred_score_map = {"pred_masks": score_map}
            
        # (b) Decoder path
        q = q.transpose(0,1) 
        q = q.permute(0, 2, 1, 3).contiguous() 
        for idx, decoder_ in enumerate(self.decoder):
            q_, attn_ = decoder_(q, lateral.transpose(0, 1))
            for q, attn in zip(q_, attn_):
                attn = attn.transpose(-1, -2) 
                attn = self.d3_to_d4(attn)
                maps_size.append(attn.size()[-2:])
                qs.append(q.transpose(0, 1))
                attns.append(attn)
                
            ## base output
            d_input = attns[-1]
            
            ## AttentionOT
            for attn in attns[:-1]:
                d_input *= torch.sigmoid(attn).pow(1/(len(attn_)-1)) 
                    
        qs = torch.stack(qs, dim=0)

        # Upsample
        pred = F.interpolate(d_input, size=(self.image_size, self.image_size),mode='bilinear', align_corners=False)
        out = {"pred_masks": pred}
        
        if self.training:
            pass
        else:
            # (c) Inference with ensemble
            balancing_factor  = 0.5
            pred = balancing_factor * pred + (1-balancing_factor) * score_map
            out = {"pred_masks": pred}
            if self_training:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx)
            else:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.1)
            return out["pred"]         
                 
        return out, pred_score_map

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            for a in outputs_seg_masks[:-1]
        ]

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def get_qs(self, q, cls):
        K, N, dim = q.shape
        bs, dim = cls.shape
        q = q.unsqueeze(0)
        q = q.expand(bs, -1, -1, -1)
        q1 = torch.einsum("bd,bcnd->bcnd", cls, q)
        q_ = torch.concat((q1, q), dim=-1)
        return q_


    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)

            loss = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index = self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss