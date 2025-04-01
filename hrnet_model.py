import torch
from torch import nn
import timm
from mvaggregate import MVAggregate
from utils import batch_tensor, unbatch_tensor

class HRNetVideoAdapter(nn.Module):
    """
    Adapter class to make HRNet work with video data.
    This wraps the HRNet model to handle the temporal dimension and reduce feature dimension.
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
        # x shape: [B, C, H, W]
        # Process the frame with HRNet
        frame_features = self.hrnet_model(x)  # [B, 2048]
        
        # Reduce feature dimension
        reduced_features = self.feature_reducer(frame_features)  # [B, 512]
        
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
        return self.mvnetwork(mvimages)