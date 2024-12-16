# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import typing
import torch.nn.functional as F
import torch.nn as nn
from torch import nn
from rvc.lib.algorithm.custom_discriminators.vocoder_blocks import *

import torch.utils.checkpoint as cp

from einops import rearrange
import torchaudio.transforms as T

from typing import List, Tuple
from nnAudio import features
import logging
logging.getLogger("nnAudio").setLevel(logging.ERROR)


LRELU_SLOPE = 0.1


class DiscriminatorCQT(nn.Module):
    def __init__(
        self,
        filters,
        max_filters,
        filters_scale,
        dilations,
        in_channels,
        out_channels,
        hop_lengths,
        n_octaves,
        bins_per_octaves,
        sample_rate,
        cqtd_normalize_volume=False,
        is_san=False,
    ):
        super().__init__()
        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = (3, 9)
        self.dilations = dilations
        self.stride = (1, 2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fs = sample_rate
        self.hop_length = hop_lengths
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octaves
        self.cqtd_normalize_volume = cqtd_normalize_volume

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                NormConv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            NormConv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(self.kernel_size, (dilation, 1)),
                    norm="weight_norm",
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
                norm="weight_norm",
            )
        )

        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            norm="weight_norm",
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=LRELU_SLOPE, inplace=False)
        self.resample = T.Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = self.cqtd_normalize_volume
        if self.cqtd_normalize_volume:
            print(
                f"[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!"
            )

    def forward(self, x: torch.tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        # Wrap convolutions and activations in checkpointing
        for i, l in enumerate(self.convs):
            # Define the forward pass for each layer
            def layer_forward(layer, activation, input):
                return activation(layer(input))

            # Apply checkpointing for the convolution layers and activation functions
            latent_z = cp.checkpoint(layer_forward, l, self.activation, latent_z, use_reentrant=False)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap

class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(
        self,
        filters=64, 
        max_filters=1024,
        filters_scale=1,
        dilations=[1, 2, 4],
        in_channels=1,
        out_channels=1,
        hop_lengths=[512, 256, 256], # [512, 256, 256],
        n_octaves=[9, 9, 9],
        bins_per_octaves=[24, 36, 48],
        sample_rate=48000,
    ):
        super().__init__()

        # Debugging: print hop_lengths and check each value
#        for i, hop_length in enumerate(hop_lengths):
#            print(f"Discriminator {i} - hop_length: {hop_length}")
#            if hop_length <= 0:
#                raise ValueError(f"Invalid hop_length {hop_length} at index {i}. It must be a positive integer.")

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    filters=filters,
                    max_filters=max_filters,
                    filters_scale=filters_scale,
                    dilations=dilations,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hop_lengths=hop_lengths[i],
                    n_octaves=n_octaves[i],
                    sample_rate=sample_rate,
                    bins_per_octaves=bins_per_octaves[i],
                )
                for i in range(len(hop_lengths))
            ]
        )

    def forward(
        self, y, y_hat
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs