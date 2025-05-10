import torch
import torch.nn.functional as F
from torch_dct import dct, idct

def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    return 2 * sum(
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    )


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.

    Args:
        disc_real_outputs (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_generated_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    loss = 0
    # r_losses = []
    # g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)

        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())
        loss += r_loss + g_loss

    return loss # , r_losses, g_losses


def generator_loss(disc_outputs):
    """
    Compute the generator loss based on discriminator outputs.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        # gen_losses.append(l.item())
        loss += l

    return loss, gen_losses


def discriminator_loss_scaled(disc_real, disc_fake, scale=1.0):
    loss = 0
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = torch.mean((1 - d_real) ** 2)
        fake_loss = torch.mean(d_fake**2)
        _loss = real_loss + fake_loss
        loss += _loss if i < len(disc_real) / 2 else scale * _loss
    return loss, None, None


def generator_loss_scaled(disc_outputs, scale=1.0):
    loss = 0
    for i, d_fake in enumerate(disc_outputs):
        d_fake = d_fake.float()
        _loss = torch.mean((1 - d_fake) ** 2)
        loss += _loss if i < len(disc_outputs) / 2 else scale * _loss
    return loss, None, None

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Latent variable z_p [b, h, t_t].
        logs_q (torch.Tensor): Log variance of q [b, h, t_t].
        m_p (torch.Tensor): Mean of p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()

    return loss


def tm_loss(
    gen_audio,
    real_audio,
    n_fft=2048,
    hop_length=480,
    win_length=2048,
    lifter_cutoff=30,
    eps=1e-8
):
    window = torch.hann_window(win_length).to(gen_audio.device)

    real_spec = torch.stft(real_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    gen_spec  = torch.stft(gen_audio,  n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

    real_mag = real_spec.abs()
    gen_mag  = gen_spec.abs()

    # Move frequency axis to last (B, T, F)
    real_mag = real_mag.transpose(1, 2)
    gen_mag  = gen_mag.transpose(1, 2)

    # Apply DCT along the last dim (freq)
    real_ceps = dct(torch.log(real_mag + eps), norm='ortho')
    gen_ceps  = dct(torch.log(gen_mag  + eps), norm='ortho')

    # Zero out higher quefrency coefficients
    real_ceps[..., lifter_cutoff:] = 0
    gen_ceps[...,  lifter_cutoff:] = 0

    # Inverse DCT to get the smoothed envelope
    real_env = torch.exp(idct(real_ceps, norm='ortho'))
    gen_env  = torch.exp(idct(gen_ceps,  norm='ortho'))

    # Restore original shape (B, F, T)
    real_env = real_env.transpose(1, 2)
    gen_env  = gen_env.transpose(1, 2)

    return F.mse_loss(gen_env, real_env)
