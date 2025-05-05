from tqdm import tqdm
from rendering import rendering
import torch
import torch.nn.functional as F
import math

def mse2psnr(mse):
    return -10. * torch.log10(mse)

def training(model, optimizer, scheduler,
             tn, tf, nb_bins,
             nb_epochs, data_loader,
             device='cpu'):
    # flat per‐batch losses
    batch_losses = []

    # per‐epoch metrics
    epoch_losses = []
    epoch_psnrs  = []

    for epoch in range(1, nb_epochs+1):
        this_epoch_losses = []
        this_epoch_psnrs  = []

        for batch in tqdm(data_loader, desc=f"Epoch {epoch}/{nb_epochs}"):
            o      = batch[:, :3].to(device)
            d      = batch[:, 3:6].to(device)
            target = batch[:, 6:].to(device)

            pred = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_val = loss.detach().cpu().item()
            psnr   = mse2psnr(loss.detach()).cpu().item()

            # record
            batch_losses.append(mse_val)
            this_epoch_losses.append(mse_val)
            this_epoch_psnrs.append(psnr)

        # end of epoch → aggregate
        avg_loss = sum(this_epoch_losses) / len(this_epoch_losses)
        avg_psnr = sum(this_epoch_psnrs)  / len(this_epoch_psnrs)

        epoch_losses.append(avg_loss)
        epoch_psnrs.append(avg_psnr)

        scheduler.step()
        torch.save(model.cpu(), f'./checkpoints/model_nerf_epoch{epoch}.pth')
        model.to(device)

        print(f"[Epoch {epoch}]  Avg Loss: {avg_loss:.4f}  |  Avg PSNR: {avg_psnr:.2f} dB")

    return batch_losses, epoch_losses, epoch_psnrs
