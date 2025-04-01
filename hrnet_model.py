import torch
from torch import nn
import timm
from mvaggregate import MVAggregate

class HRNetVideoAdapter(nn.Module):
    """
    Adapter class to make HRNet work with video data.
    This wraps the HRNet model to handle the temporal dimension.
    """
    def __init__(self, hrnet_model):
        super().__init__()
        self.hrnet_model = hrnet_model
        
        # Feature dimension reduction layer
        self.feature_reducer = nn.Sequential(
            nn.Linear(2048, 512),  # Assuming HRNet W18 outputs 2048 features
            nn.ReLU()
        )
    
    def forward(self, x):
        # Potential input shapes:
        # 1. [B, V, C, D, H, W] - batched multi-view video
        # 2. [B, V, C, H, W] - batched multi-view frames
        # 3. [B, C, D, H, W] - batched video
        # 4. [B, C, H, W] - batched single frame
        
        # Normalize input to 4D tensor [B, C, H, W]
        if len(x.shape) == 6:
            # [B, V, C, D, H, W] - multi-view video
            B, V, C, D, H, W = x.shape
            # Select middle frame for each view
            x = x[:, :, :, D//2, :, :]  # [B, V, C, H, W]
        
        if len(x.shape) == 5:
            # [B, V, C, H, W] - multi-view frames
            B, V, C, H, W = x.shape
            # Reshape to process all views
            x = x.view(B * V, C, H, W)
        
        elif len(x.shape) == 4:
            # [B, C, H, W] or [B, C, D, H, W]
            if x.shape[1] > 3:
                # If more than 3 channels, assume it's a video with depth
                D, H, W = x.shape[2], x.shape[3], x.shape[4]
                x = x[:, :, D//2, :, :]  # Select middle frame
        
        # Process the frame with HRNet
        frame_features = self.hrnet_model(x)  # [B, 2048]
        
        # Reduce feature dimension
        reduced_features = self.feature_reducer(frame_features)  # [B, 512]
        
        # Reshape back to original batch structure if needed
        if len(x.shape) >= 5:
            reduced_features = reduced_features.view(B, V, -1)  # [B, V, 512]
        
        return reduced_features

class HRNetMVNetwork(torch.nn.Module):
    def __init__(self, agr_type='max', lifting_net=torch.nn.Sequential()):
        super().__init__()
        
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        
        # HRNet feature dimension
        self.feat_dim = 512
        
        # Load HRNet from timm
        base_model = timm.create_model('hrnet_w18', pretrained=True)
        
        # Replace the classifier head with an identity operation
        self.backbone = base_model
        self.backbone.classifier = nn.Identity()
        
        # Create the video adapter for HRNet
        self.video_model = HRNetVideoAdapter(self.backbone)
        
        # Create the MVAggregate module
        self.mvnetwork = MVAggregate(
            model=self.video_model,
            agr_type=self.agr_type, 
            feat_dim=self.feat_dim, 
            lifting_net=self.lifting_net,
        )
        
    def forward(self, mvimages):
        # Handle different possible input shapes
        if len(mvimages.shape) == 5:
            # [B, V, C, H, W] or [B, V, C, T, H, W]
            B, V, C, *rest = mvimages.shape
            if len(rest) == 3:  # [B, V, C, T, H, W]
                mvimages = mvimages[:, :, :, mvimages.shape[3]//2, :, :]  # Select middle frame
        
        return self.mvnetwork(mvimages)