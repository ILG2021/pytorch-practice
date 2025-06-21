import torch
from torch import nn

from llm.model import Transformer, SinPositionEncoding


class VisionTransformer(nn.Module):
    def __init__(self, in_channel, image_size, patch_size, embed_size, num_layers):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channel, embed_size, patch_size,
                                     patch_size)  # 输出 embed_size, image_size / patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.transformer = Transformer(embed_size, num_layers)
        self.position_encoding = SinPositionEncoding(embed_size, 1 + (image_size // patch_size) ** 2)

    # b c h w
    def forward(self, x: torch.Tensor, return_attention=False):
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # b d h', w'
        x = x.flatten(2).transpose(1, 2)  # b n d
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.position_encoding(x)
        x, attention_scores = self.transformer(x)
        if return_attention:
            return x, attention_scores
        else:
            return x


class VITClassifier(nn.Module):
    def __init__(self, in_channel, image_size, patch_size, embed_size, num_layers, num_classes):
        super().__init__()
        self.vit = VisionTransformer(in_channel, image_size, patch_size, embed_size, num_layers)
        self.linear = nn.Linear(embed_size, num_classes)

    def forward(self, x, return_attention_score=False):
        if return_attention_score:
            x, score = self.vit(x, return_attention_score)  # b n d
            return self.linear(x[:, 0, :]), score  # x[:, 0, :] 第一个token b d
        else:
            x = self.vit(x)
            return self.linear(x[:, 0, :])
