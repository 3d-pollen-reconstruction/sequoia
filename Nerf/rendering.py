import torch


def compute_accumulated_transmittance(betas):
    # Ensure betas is float32 if it might not be? (Optional, depends on input)
    # betas = betas.type(torch.float32)
    accumulated_transmittance = torch.cumprod(betas, 1)
    # Explicitly create ones as float32
    ones_tensor = torch.ones(
        accumulated_transmittance.shape[0],
        1,
        dtype=torch.float32,
        device=accumulated_transmittance.device,
    )  # Added dtype
    return torch.cat((ones_tensor, accumulated_transmittance[:, :-1]), dim=1)


def rendering(
    model, rays_o, rays_d, tn, tf, nb_bins=100, device="cpu", white_bckgr=True
):
    # Ensure inputs are float32 (Important if not guaranteed by caller)
    rays_o = rays_o.type(torch.float32)
    rays_d = rays_d.type(torch.float32)

    t = torch.linspace(tn, tf, nb_bins, dtype=torch.float32).to(
        device
    )  # OK: dtype specified

    # Explicitly create delta component as float32
    delta_last = torch.tensor([1e10], dtype=torch.float32, device=device)  # Added dtype
    delta = torch.cat((t[1:] - t[:-1], delta_last))

    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(
        1
    )  # [nb_rays, nb_bins, 3]

    # Ensure model inputs/outputs are float32 (Model dependent)
    # Assuming model.intersect takes float32 and returns float32
    colors, density = model.intersect(
        x.reshape(-1, 3),
        rays_d.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3),
    )

    colors = colors.reshape((x.shape[0], nb_bins, 3))  # [nb_rays, nb_bins, 3]
    density = density.reshape(
        (x.shape[0], nb_bins)
    )  # Assuming density is float32 out of model

    # Calculation depends on density and delta dtypes
    alpha = (
        1 - torch.exp(-density * delta.unsqueeze(0))
    )  # [nb_rays, nb_bins] # Note: unsqueezed delta might need expansion if not broadcasting correctly

    # Calculation depends on alpha dtype
    weights = compute_accumulated_transmittance(1 - alpha) * alpha  # [nb_rays, nb_bins]

    # Final calculation depends on weights and colors dtypes
    if white_bckgr:
        c = (weights.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]
        weight_sum = weights.sum(-1)  # [nb_rays]
        # Ensure background calculation is float32
        return (
            c + 1 - weight_sum.unsqueeze(-1).type(torch.float32)
        )  # Added type just in case
    else:
        c = (weights.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]

    return c.type(torch.float32)  # Ensure final output is float32
