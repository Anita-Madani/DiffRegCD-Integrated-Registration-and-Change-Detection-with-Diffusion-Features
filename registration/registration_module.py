import torch
import torch.nn as nn
from registration.decoder_reg import TransformerDecoder
from registration.transformer.layers.block import Block
from registration.transformer.layers.attention import MemEffAttention
from registration.utils import CosKernel, GP, cls_to_flow_refine


class RegistrationModule(nn.Module):
    def __init__(self):
        super().__init__()
        decoder_dim = 1024 + 512
        out_dim = 32 * 32 + 1
        blocks = nn.Sequential(*[
            Block(decoder_dim, 8, attn_class=MemEffAttention)
            for _ in range(5)
        ])
        self.decoder = TransformerDecoder(
            blocks, decoder_dim, out_dim,
            is_classifier=True, amp=True, pos_enc=True
        )

        self.gp = GP(
            CosKernel, T=0.2, learn_temperature=False,
            only_attention=False, gp_dim=512, basis="fourier", no_cov=True
        )


        ##Added 
        self.temperature = nn.Parameter(torch.tensor(0.7))

    def forward(self, f0, f1):
        # Automatically move everything to match f0's device
        device = f0.device
        self.decoder = self.decoder.to(device)
        self.gp = self.gp.to(device)

        for name, module in self.decoder.named_modules():
            for param in module.parameters(recurse=False):
                if param.device != device:
                    param.data = param.data.to(device)
            for buffer_name, buffer in module._buffers.items():
                if buffer is not None and buffer.device != device:
                    module._buffers[buffer_name] = buffer.to(device)

        gp_post = self.gp(f0, f1)
        logits, _ = self.decoder(gp_post, f0, None, 16)
#        print("%%%%%%%% shape of the logits in the reg %%%%%%%", logits.shape)

        ##ADDED##
        T = self.temperature.clamp(0.3, 1.5)
        logits = logits / T

        
        pred_flow = cls_to_flow_refine(logits)
#        print(" &&&&& shape of the flow in the reg &&&&&&& ", pred_flow.shape)
        return pred_flow, logits
