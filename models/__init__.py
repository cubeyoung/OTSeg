from models.segmentor.zegclip import OTSeg

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from models.decode_heads.decode_seg import ATMSingleHeadSeg

from models.losses.atm_loss import SegLossPlus

from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
# from configs._base_.datasets.dataloader.context import ZeroPascalContextDataset59, ZeroPascalContextDataset459
from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
from configs._base_.datasets.dataloader.ade import ZeroADE20KDataset847, ZeroADE20KDataset
