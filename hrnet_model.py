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
        # Handle different possible input shapes
        if len(x.shape) == 5:
            # Input shape: [B, V, C, H, W]
            B, V, C, H, W = x.shape
            
            # Reshape to process frames individually
            x_reshaped = x.view(B * V, C, H, W)
        elif len(x.shape) == 4:
            # Input shape: [B, C, H, W]
            B, C, H, W = x.shape
            V = 1
            x_reshaped = x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Process the frame with HRNet
        frame_features = self.hrnet_model(x_reshaped)  # [B*V, 2048]
        
        # Reduce feature dimension
        reduced_features = self.feature_reducer(frame_features)  # [B*V, 512]
        
        # Reshape back to original batch structure
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
        # Ensure input is expected shape [B, V, C, D, H, W]
        return self.mvnetwork(mvimages)