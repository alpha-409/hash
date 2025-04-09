# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import kornia # For Canny, Gaussian Blur etc.
import time # For checking SRD speed

# --- Utility Functions / Layers ---

class SignWithStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # Save input for potential clamped STE
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass gradient through unchanged for inputs in [-1, 1]
        # This version clamps gradients outside [-1, 1] which can sometimes help
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1.0] = 0 # Zero gradient for inputs far from {-1, 1}
        return grad_input

sign_ste = SignWithStraightThrough.apply

# --- SRD Module ---

class SRD(nn.Module):
    def __init__(self, edge_sigma=1.5, edge_kernel_size=11, color_bins=16, use_cuda=True):
        super().__init__()
        self.edge_sigma = edge_sigma
        self.edge_kernel_size = edge_kernel_size
        self.color_bins = color_bins
        self.epsilon = 1e-9
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # Precompute Gaussian kernel if possible, or use Kornia's layer
        self.gaussian_blur = kornia.filters.GaussianBlur2d(
            (self.edge_kernel_size, self.edge_kernel_size),
            (self.edge_sigma, self.edge_sigma)
        ).to(self.device)

        # Predefined mean and std for potential denormalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, x_norm):
        """
        Args:
            x_norm (torch.Tensor): Input NORMALIZED image tensor (B, C, H, W).
        Returns:
            torch.Tensor: Saliency map W_s (B, 1, H, W).
        """
        # start_time = time.time()
        x_norm = x_norm.to(self.device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

        B, C, H, W = x_norm.shape

        # --- Denormalize for processing ---
        x = x_norm * self.std + self.mean
        x = x.clamp(0.0, 1.0) # Ensure values are in [0, 1]

        # --- 1. Edge Density ---
        if C == 3:
            gray = kornia.color.rgb_to_grayscale(x)
        else:
            gray = x

        # Kornia's Canny can be slow, especially on CPU. Thresholds might need tuning.
        try:
             # Use kornia directly
             _, edges = kornia.filters.canny(gray, low_threshold=0.1, high_threshold=0.2, kernel_size=5) # Example thresholds/kernel
             edges = edges.float()
        except Exception as e:
             print(f"Kornia Canny error: {e}. Using alternative (potentially slower).")
             # Fallback or simplified edge detection if kornia fails
             # Example: Sobel filter (less accurate than Canny)
             sobel = kornia.filters.Sobel().to(self.device)
             edges = sobel(gray).abs().mean(dim=1, keepdim=True) # Average magnitude over channels
             # Simple thresholding
             edge_threshold = torch.quantile(edges.view(B, -1), 0.8, dim=1, keepdim=True).view(B, 1, 1, 1)
             edges = (edges > edge_threshold).float()


        edge_density = self.gaussian_blur(edges)

        # --- 2. Color Sparsity ---
        # Quantize colors
        quantized_colors = (x * (self.color_bins - 1)).round().long()

        if C == 3:
            combined_indices = quantized_colors[:, 0, :, :] * (self.color_bins ** 2) + \
                               quantized_colors[:, 1, :, :] * self.color_bins + \
                               quantized_colors[:, 2, :, :]
            num_total_bins = self.color_bins ** 3
        else: # Grayscale
             combined_indices = quantized_colors[:, 0, :, :]
             num_total_bins = self.color_bins

        color_sparsity = torch.zeros_like(gray) # (B, 1, H, W)
        for i in range(B):
            img_indices = combined_indices[i].view(-1)
            # Use minlength to ensure consistent histogram size
            counts = torch.bincount(img_indices, minlength=num_total_bins)
            total_pixels = H * W
            probabilities = counts.float() / total_pixels
            # Ensure probabilities tensor is on the correct device before indexing
            probabilities = probabilities.to(self.device)
            pixel_probs = probabilities[combined_indices[i]] # Indexing happens here
            img_sparsity = -torch.log(pixel_probs + self.epsilon)
            color_sparsity[i, 0, :, :] = img_sparsity

        # --- 3. Fusion ---
        def normalize_map(feat_map):
            B, _, _, _ = feat_map.shape
            min_vals = torch.min(feat_map.view(B, -1), dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            max_vals = torch.max(feat_map.view(B, -1), dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            range_vals = max_vals - min_vals
            normalized = (feat_map - min_vals) / (range_vals + self.epsilon)
            return normalized

        # Normalize per image
        norm_edge_density = normalize_map(edge_density)
        norm_color_sparsity = normalize_map(color_sparsity)

        fused_map = self.alpha * norm_edge_density + self.beta * norm_color_sparsity
        W_s = torch.sigmoid(fused_map)

        # print(f"SRD time: {time.time() - start_time:.4f}s")
        return W_s.to(x_norm.device) # Ensure output is on the same device as input

# --- DFE Module ---

class DetailPath(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_layers=4):
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_channels = out_channels
        self.path = nn.Sequential(*layers)

    def forward(self, x):
        return self.path(x)

class DynamicAttentionGate(nn.Module):
    def __init__(self, detail_channels, semantic_channels):
        super().__init__()
        gate_channels = (detail_channels + semantic_channels) // 2
        self.conv_gate = nn.Sequential(
            nn.Conv2d(detail_channels + semantic_channels, gate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_detail, f_semantic_upsampled):
        combined = torch.cat([f_detail, f_semantic_upsampled], dim=1)
        gate = self.conv_gate(combined)
        return gate

class DFE(nn.Module):
    def __init__(self, detail_channels=64, backbone_name='resnet18', pretrained=True):
        super().__init__()
        self.detail_path = DetailPath(in_channels=3, out_channels=detail_channels)

        # Load backbone and determine semantic channels
        if backbone_name.startswith('resnet'):
            if backbone_name == 'resnet18':
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
                self.semantic_path = nn.Sequential(*list(backbone.children())[:-3]) # Output of layer3
                self.semantic_channels = 256
            elif backbone_name == 'resnet34':
                 backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
                 self.semantic_path = nn.Sequential(*list(backbone.children())[:-3])
                 self.semantic_channels = 256
            elif backbone_name == 'resnet50':
                 backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
                 self.semantic_path = nn.Sequential(*list(backbone.children())[:-3])
                 self.semantic_channels = 1024 # ResNet50 layer3 output
            else:
                 raise ValueError(f"Unsupported ResNet variant: {backbone_name}")
        # Add other backbones like VGG etc. if needed
        # elif backbone_name.startswith('vgg'):
            # backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            # Select features, e.g., from conv4_3
            # self.semantic_path = backbone.features[:23] # Example: up to pool4
            # self.semantic_channels = 512 # VGG16 pool4 output
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # self.fusion_gate = DynamicAttentionGate(detail_channels, self.semantic_channels)
        self.fusion_gate = DynamicAttentionGate(self.semantic_channels, self.semantic_channels)

        # Add projection layer if detail and semantic channels differ, needed for fusion
        self.fused_channels = self.semantic_channels # Fusion result will have semantic channels
        if detail_channels != self.semantic_channels:
            self.detail_proj = nn.Conv2d(detail_channels, self.semantic_channels, kernel_size=1, bias=False)
        else:
            self.detail_proj = nn.Identity()


    def forward(self, x, W_s):
        # Apply Saliency Guidance
        x_guided = x * W_s

        # Detail Path
        f_detail = self.detail_path(x_guided)

        # Semantic Path
        f_semantic = self.semantic_path(x_guided)

        # Upsample Semantic Features
        target_size = f_detail.shape[2:]
        f_semantic_upsampled = F.interpolate(f_semantic, size=target_size, mode='bilinear', align_corners=False)

        # Project detail features if necessary
        f_detail_proj = self.detail_proj(f_detail)

        # Attention Gate
        attention_gate = self.fusion_gate(f_detail_proj, f_semantic_upsampled) # Use projected detail

        # Weighted Fusion
        f_fused = attention_gate * f_semantic_upsampled + (1 - attention_gate) * f_detail_proj

        return f_fused

# --- CP-Hash Layer ---

class CPHashLayer(nn.Module):
    def __init__(self, in_channels, hash_bits=128, cp_rank=32):
        super().__init__()
        self.in_channels = in_channels
        self.hash_bits = hash_bits
        self.cp_rank = cp_rank

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        # MLP approximating factor generation and combination
        # Simplified: Pool -> FC -> BN -> ReLU -> FC (to hash_bits)
        factor_hidden_dim = max(in_channels // 2, hash_bits * 2) # Example heuristic

        self.hash_mlp = nn.Sequential(
            nn.Linear(in_channels, factor_hidden_dim),
            nn.BatchNorm1d(factor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(factor_hidden_dim, hash_bits)
            # Optional Tanh: nn.Tanh() # If using Tanh, loss might need adjustment
        )

    def forward(self, f_fused):
        pooled = self.pool(f_fused)
        vector = self.flatten(pooled)
        tilde_H = self.hash_mlp(vector) # Pre-binary hash code
        H = sign_ste(tilde_H)          # Binary hash code {-1, 1}
        return H, tilde_H

# --- Full Model ---

class ScreenHashNet(nn.Module):
    def __init__(self,
                 hash_bits=128,
                 cp_rank=32,
                 srd_sigma=1.5,
                 srd_kernel_size=11,
                 srd_color_bins=16,
                 dfe_detail_channels=64,
                 dfe_backbone='resnet18',
                 dfe_pretrained=True,
                 use_cuda=True):
        super().__init__()
        self.srd = SRD(edge_sigma=srd_sigma, edge_kernel_size=srd_kernel_size,
                       color_bins=srd_color_bins, use_cuda=use_cuda)
        self.dfe = DFE(detail_channels=dfe_detail_channels,
                       backbone_name=dfe_backbone,
                       pretrained=dfe_pretrained)
        self.cp_hash_layer = CPHashLayer(in_channels=self.dfe.fused_channels,
                                         hash_bits=hash_bits,
                                         cp_rank=cp_rank)

    def forward(self, x):
        W_s = self.srd(x)
        f_fused = self.dfe(x, W_s)
        H, tilde_H = self.cp_hash_layer(f_fused)
        return H, tilde_H

# --- Loss Function ---

class HashingLoss(nn.Module):
    def __init__(self, lambda_sim=1.0, lambda_quant=0.5, lambda_balance=0.1, sim_margin=0.5):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_quant = lambda_quant
        self.lambda_balance = lambda_balance
        self.sim_margin = sim_margin # Margin for dot product similarity

    def forward(self, H, tilde_H, S):
        """
        Args:
            H (torch.Tensor): Binary hash codes (B, L), values {-1, 1}.
            tilde_H (torch.Tensor): Pre-binary hash codes (B, L).
            S (torch.Tensor): Similarity matrix (B, B). S_ij=1 (similar), S_ij=-1 (dissimilar), S_ij=0 (ignore).
        """
        B, L = H.shape
        device = H.device

        # --- 1. Similarity Loss (Pairwise Dot Product Based) ---
        dot_products = torch.matmul(H, H.t()) # Shape (B, B)
        # Loss: max(0, margin - S_ij * dot_product_ij)^2
        # Where dot_product is implicitly scaled by L. Let's normalize by sqrt(L) or L?
        # Let's try normalized dot product: dot(H1, H2) / L ranges approx [-1, 1]
        norm_dot_products = dot_products / L
        loss_sim_pairs = torch.pow(F.relu(self.sim_margin - S * norm_dot_products), 2)

        # Mask out diagonal and ignored pairs
        mask = (S != 0).float().to(device)
        diag_mask = (1.0 - torch.eye(B, device=device))
        loss_sim_pairs = loss_sim_pairs * mask * diag_mask

        num_valid_pairs = torch.sum(mask * diag_mask)
        loss_sim = torch.sum(loss_sim_pairs) / (num_valid_pairs + 1e-9) if num_valid_pairs > 0 else torch.tensor(0.0).to(device)


        # --- 2. Quantization Loss ---
        loss_quant = torch.mean(torch.pow(torch.abs(tilde_H) - 1.0, 2))

        # --- 3. Bit Balance Loss ---
        # Encourage mean of pre-binary values towards 0 for each bit
        bit_means_tilde = torch.mean(tilde_H, dim=0)
        loss_balance = torch.mean(torch.pow(bit_means_tilde, 2)) # Use mean instead of sum for scale invariance

        # --- Total Loss ---
        total_loss = (self.lambda_sim * loss_sim +
                      self.lambda_quant * loss_quant +
                      self.lambda_balance * loss_balance)

        loss_dict = {
            'total_loss': total_loss,
            'loss_sim': loss_sim,
            'loss_quant': loss_quant,
            'loss_balance': loss_balance
        }

        return total_loss, loss_dict