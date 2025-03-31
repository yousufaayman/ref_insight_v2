import torch
from torch import nn
import timm
from mvaggregate import MVAggregate
from utils import batch_tensor, unbatch_tensor


class HRNetVideoAdapter(nn.Module):
    """
    Adapter class to make HRNet work with video data.
    This wraps the HRNet model to handle the temporal dimension.
    """
    def __init__(self, hrnet_model):
        super().__init__()
        self.hrnet_model = hrnet_model
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
    
    def forward(self, x):
        # x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Process each frame with HRNet
        features = []
        for t in range(T):
            # Extract frame
            frame = x[:, :, t, :, :]  # [B, C, H, W]
            
            # Forward through HRNet
            frame_features = self.hrnet_model(frame)  # [B, feat_dim]
            
            # Collect features
            features.append(frame_features)
        
        # Stack along temporal dimension
        features = torch.stack(features, dim=2)  # [B, feat_dim, T]
        
        # Apply temporal pooling to get video-level features
        features = self.temporal_pool(features).squeeze(2)  # [B, feat_dim]
        
        return features


class HRNetMVNetwork(torch.nn.Module):
    def __init__(self, agr_type='max', lifting_net=torch.nn.Sequential()):
        super().__init__()
        
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        
        # HRNet feature dimension
        self.feat_dim = 2048
        
        # Load HRNet from timm
        base_model = timm.create_model('hrnet_w64', pretrained=True)
        
        # Remove the classifier to get features
        modules = list(base_model.children())[:-1]
        backbone = nn.Sequential(*modules)
        
        # Create the video adapter for HRNet
        self.video_model = HRNetVideoAdapter(backbone)
        
        # Create the MVAggregate module
        self.mvnetwork = MVAggregate(
            model=self.video_model,
            agr_type=self.agr_type, 
            feat_dim=self.feat_dim, 
            lifting_net=self.lifting_net,
        )
        
    def forward(self, mvimages):
        return self.mvnetwork(mvimages)