from tqdm import tqdm
from rendering import rendering
import torch

def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, device='cpu'):
    
    training_loss = []
    for epoch in (range(nb_epochs)):
        for batch in tqdm(data_loader):
            o = batch[:, :3].type(torch.float32).to(device)
            d = batch[:, 3:6].type(torch.float32).to(device)
            target = batch[:, 6:].type(torch.float32).to(device)
            
            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
            
            loss = ((prediction - target)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.detach().item())
            
        scheduler.step()
        
        torch.save(model.cpu(), 'model_nerf')
        model.to(device)
        
    return training_loss