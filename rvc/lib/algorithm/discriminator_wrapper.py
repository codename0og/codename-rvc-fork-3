import torch

from rvc.lib.algorithm.custom_discriminators.__init__ import MultiScaleSTFTDiscriminator
from rvc.lib.algorithm.custom_discriminators.__init__ import MultiScaleSubbandCQTDiscriminator
from rvc.lib.algorithm.custom_discriminators.__init__ import MultiPeriodDiscriminator

class CombinedDiscriminator(torch.nn.Module):
    def __init__(self, discriminators):
        super().__init__()
        self.discriminators = torch.nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)
            
            # Check the type of the discriminator and handle accordingly
            #if isinstance(disc, MultiScaleSTFTDiscriminator):
            #    # For STFT discriminator, pass one input at a time
            #    y_d_r, fmap_r = disc(y)  # Real data
            #    y_d_g, fmap_g = disc(y_hat)  # Generated data
            #else:
            #    # For other discriminators (like MSSBCQT), pass both y and y_hat
            #    y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)

            # Collect results
            y_d_rs.extend(y_d_r)
            fmap_rs.extend(fmap_r)
            y_d_gs.extend(y_d_g)
            fmap_gs.extend(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
