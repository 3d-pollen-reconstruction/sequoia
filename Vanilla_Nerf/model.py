import torch
import torch.nn as nn

class Voxels(nn.Module):
    
    def __init__(self, nb_voxels=100, scale=1, device='cpu'):
        super(Voxels, self).__init__()
        
        self.voxels = torch.nn.Parameter(torch.rand((nb_voxels, nb_voxels, nb_voxels, 4), 
                                                    device=device, requires_grad=True))
        
        self.nb_voxels = nb_voxels
        self.device = device
        self.scale = scale
        
    def forward(self, xyz, d):
        
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        
        cond = (x.abs() < (self.scale / 2)) & (y.abs() < (self.scale / 2)) & (z.abs() < (self.scale / 2))
        
        indx = (x[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indy = (y[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indz = (z[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        
        colors_and_densities = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        colors_and_densities[cond, :3] = self.voxels[indx, indy, indz, :3]
        colors_and_densities[cond, -1] = self.voxels[indx, indy, indz, -1]
         
        return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(colors_and_densities[:, -1:])
        
    
    def intersect(self, x, d):
        return self.forward(x, d)
    
    
class Nerf(nn.Module):
    
    def __init__(self, Lpos=10, Ldir=4, hidden_dim=256):
        super(Nerf, self).__init__()
        
        # REDUCE positional encoding frequencies to prevent exploding gradients
        self.Lpos = min(Lpos, 8)  # Cap at 8 instead of 10
        self.Ldir = min(Ldir, 3)  # Cap at 3 instead of 4
        
        pos_input_dim = self.Lpos * 6 + 3
        dir_input_dim = self.Ldir * 6 + 3
        
        # First block with better initialization
        self.block1 = nn.Sequential(
            nn.Linear(pos_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Second block with skip connection
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)  # +1 for density
        )
        
        # RGB head
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
        
        # CRITICAL: Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to prevent collapse"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization for stability
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Special initialization for density layer (last layer of block2)
        with torch.no_grad():
            final_layer = self.block2[-1]
            # Initialize density bias to small positive value
            final_layer.bias[-1] = 0.1
            # Initialize density weights smaller
            final_layer.weight[-1, :] *= 0.1
        
    def positional_encoding(self, x, L):
        """Improved positional encoding with frequency clamping"""
        out = [x]
        for j in range(L):
            freq = 2.0 ** j
            # Clamp frequencies to prevent explosion
            freq = min(freq, 512.0)  # Cap maximum frequency
            
            sin_comp = torch.sin(freq * x)
            cos_comp = torch.cos(freq * x)
            
            # Clamp to prevent NaN/Inf
            sin_comp = torch.clamp(sin_comp, -1.0, 1.0)
            cos_comp = torch.clamp(cos_comp, -1.0, 1.0)
            
            out.append(sin_comp)
            out.append(cos_comp)
            
        return torch.cat(out, dim=1)
        
    def forward(self, xyz, d):
        # Clamp inputs to reasonable range
        xyz = torch.clamp(xyz, -5.0, 5.0)
        d = F.normalize(d, dim=-1)  # Ensure directions are normalized
        
        # Positional encoding
        x_emb = self.positional_encoding(xyz, self.Lpos)
        d_emb = self.positional_encoding(d, self.Ldir)
        
        # Forward pass
        h = self.block1(x_emb)
        h = self.block2(torch.cat((h, x_emb), dim=1))
        
        # Extract density and features
        sigma = h[:, -1]
        h_features = h[:, :-1]
        
        # RGB prediction
        c = self.rgb_head(torch.cat((h_features, d_emb), dim=1))
        
        # Apply activations with clamping
        sigma = torch.relu(sigma)
        sigma = torch.clamp(sigma, 0.0, 10.0)  # Prevent density explosion
        
        c = torch.clamp(c, 0.0, 1.0)  # Ensure RGB in valid range
        
        return c, sigma
        
    def intersect(self, x, d):
        return self.forward(x, d)

    
import torch.nn.functional as F

class NerfPrior(nn.Module):
    def __init__(self, num_shapes, latent_dim=16, Lpos=10, Ldir=4, hidden_dim=256):
        super().__init__()
        # each shape gets a K-dim vector z
        self.z_embed = nn.Embedding(num_shapes, latent_dim)
        
        # modify input size: xyz+d plus z
        in_ch  = (Lpos*6+3)
        in_chc = (Ldir*6+3)
        self.fc_in = nn.Linear(in_ch + latent_dim, hidden_dim)
        # … then exactly your existing layers, just shifted by latent_dim
        self.block1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + in_ch + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + in_chc, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 3), nn.Sigmoid()
        )
        self.Lpos, self.Ldir = Lpos, Ldir

    def pe(self, x, L):
        out = [x]
        for i in range(L):
            out.append(torch.sin(2**i * x))
            out.append(torch.cos(2**i * x))
        return torch.cat(out, dim=-1)

    def forward(self, xyz, d, shape_id):
        """
        xyz: [B,3], d: [B,3], shape_id: scalar int or [B] long
        """
        z = self.z_embed(shape_id)                      # [B, K]
        x = self.pe(xyz, self.Lpos)                     # [B, in_ch]
        d = self.pe(d,   self.Ldir)                     # [B, in_chc]
        h = self.fc_in(torch.cat([x, z], dim=-1))
        h = self.block1(h)
        h = self.block2(torch.cat([h, x, z], dim=-1))
        
        sigma = h[..., -1]                              # [B]
        h_feat = h[..., :-1]
        c     = self.rgb_head(torch.cat([h_feat, d], dim=-1))
        return c, F.relu(sigma)

