import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from sklearn.decomposition import PCA
from PIL import Image
from torch.nn import functional as F

class ViTFeatureExtractor:
    def __init__(self, model_name="vit_b_16"):
        # Load pre-trained Vision Transformer model
        if model_name == "vit_b_16":
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_b_32":
            self.model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        elif model_name == "vit_l_16":
            self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_l_32":
            self.model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Remove classification head, keep feature extraction part
        self.feature_dim = self.model.hidden_dim
        self.model.heads = nn.Identity()
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, img):
        # If the input is a PIL image, preprocess it
        if isinstance(img, Image.Image):
            img = self.preprocess(img)
        
        # If input is a tensor but not in batch format, add batch dimension
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img.unsqueeze(0)
        
        # Move image to the device
        img = img.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            features = self.model(img)
        
        # Convert features to numpy
        features = features.cpu().numpy()
        
        # If it's a single sample, remove batch dimension
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        # Normalize features
        features = features / np.linalg.norm(features)
        
        return features


def extract_multiscale_features_vit(img, scales):
    """
    Extract features at multiple scales using ViT.
    
    Args:
        img: Input image (PIL or Tensor).
        scales: List of scale factors (e.g., [1.0, 0.75, 0.5]).
    
    Returns:
        numpy.ndarray: Feature matrix of shape (n_scales, feature_dim).
    """
    extractor = ViTFeatureExtractor()
    features_list = []
    
    for scale in scales:
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                img_tensor = img.unsqueeze(0)
            else:
                img_tensor = img
            
            # 先进行缩放，然后确保调整到224x224
            _, C, H, W = img_tensor.shape
            new_H = max(1, int(H * scale))
            new_W = max(1, int(W * scale))
            img_scaled = F.interpolate(img_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
            
            # 再次调整到ViT所需的224x224尺寸
            img_scaled = F.interpolate(img_scaled, size=(224, 224), mode='bilinear', align_corners=False)
            img_scaled = img_scaled.squeeze(0)
            
            feature = extractor.extract_features(img_scaled)
        elif isinstance(img, Image.Image):
            # 先按比例缩放
            w, h = img.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img_scaled = img.resize(new_size, Image.LANCZOS)  # 使用LANCZOS替代已弃用的ANTIALIAS
            
            # 然后调整到ViT所需的224x224尺寸
            img_scaled = img_scaled.resize((224, 224), Image.LANCZOS)
            
            feature = extractor.extract_features(img_scaled)
        else:
            raise ValueError("Unsupported image type. Only PIL.Image or torch.Tensor are supported.")
        
        if feature.ndim > 1:
            feature = feature.flatten()
        features_list.append(feature)
    
    return np.stack(features_list, axis=0)


def multiscale_vit_hash(img, hash_size=8, scales=[1.0, 0.75, 0.5]):
    """
    Generate a binary hash based on multi-scale ViT features and matrix decomposition.
    
    Args:
        img: Input image (PIL or Tensor).
        hash_size: Hash size (i.e., the edge length of the hash).
        scales: List of scale factors for multi-scale extraction.
    
    Returns:
        numpy.ndarray: Binary hash vector.
    """
    features_matrix = extract_multiscale_features_vit(img, scales)
    
    # Perform fusion and dimensionality reduction (e.g., via PCA or SVD)
    U, S, Vt = np.linalg.svd(features_matrix, full_matrices=False)
    fused_feature = Vt[0]  # Take the first principal component
    target_dim = hash_size * hash_size
    
    if fused_feature.shape[0] > target_dim:
        fused_feature = fused_feature[:target_dim]
    
    median_val = np.median(fused_feature)
    binary_hash = fused_feature > median_val
    return binary_hash


def multiscale_vit_deep(img, scales=[1.0, 0.75, 0.5]):
    """
    Generate a normalized deep feature vector based on multi-scale ViT features.
    
    Args:
        img: Input image (PIL or Tensor).
        scales: List of scale factors for multi-scale extraction.
    
    Returns:
        numpy.ndarray: Normalized feature vector.
    """
    features_matrix = extract_multiscale_features_vit(img, scales)
    
    U, S, Vt = np.linalg.svd(features_matrix, full_matrices=False)
    fused_feature = Vt[0]  # Take the first principal component
    
    norm = np.linalg.norm(fused_feature)
    if norm > 0:
        fused_feature = fused_feature / norm
    
    return fused_feature


def compute_multiscale_vit_distance(feature1, feature2):
    """
    Compute the cosine distance between two multi-scale ViT feature vectors.
    
    Args:
        feature1, feature2: Feature vectors (numpy arrays).
    
    Returns:
        float: Cosine distance (lower values mean more similarity).
    """
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    distance = 1.0 - similarity
    return distance
