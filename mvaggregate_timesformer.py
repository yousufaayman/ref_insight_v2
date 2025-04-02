import torch
from torch import nn

class WeightedAggregate(nn.Module):
    def __init__(self, model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.relu = nn.ReLU()

    def forward(self, mvimages):
        # Get features for all views
        # TimesformerVideoAdapter returns [B, V, feat_dim]
        features = self.model(mvimages)  # [B, V, feat_dim]
        
        # Apply lifting network if provided
        aux = self.lifting_net(features) if len(self.lifting_net) > 0 else features
        
        # Get shapes
        B, V, feat_dim = aux.shape

        # View attention mechanism
        aux_matmul = torch.matmul(aux, self.attention_weights)
        aux_t = aux_matmul.permute(0, 2, 1)

        prod = torch.bmm(aux_matmul, aux_t)
        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))
        final_attention_weights = torch.sum(final_attention_weights, 1)

        # Compute weighted output
        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))
        output = torch.sum(output, 1)

        # Ensure 2D output
        output = output.view(B, -1)

        return output.squeeze(), final_attention_weights

class ViewMaxAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        # Get features for all views
        # TimesformerVideoAdapter returns [B, V, feat_dim]
        features = self.model(mvimages)  # [B, V, feat_dim]
        
        # Apply lifting network if provided
        aux = self.lifting_net(features) if len(self.lifting_net) > 0 else features
        
        # Get batch size
        B = aux.shape[0]
        
        # Max pooling across views
        pooled_view = torch.max(aux, dim=1)[0].view(B, -1)
        
        return pooled_view.squeeze(), aux

class ViewAvgAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        # Get features for all views
        # TimesformerVideoAdapter returns [B, V, feat_dim]
        features = self.model(mvimages)  # [B, V, feat_dim]
        
        # Apply lifting network if provided
        aux = self.lifting_net(features) if len(self.lifting_net) > 0 else features
        
        # Get batch size
        B = aux.shape[0]
        
        # Average pooling across views
        pooled_view = torch.mean(aux, dim=1).view(B, -1)
        
        return pooled_view.squeeze(), aux

class MVAggregate(nn.Module):
    def __init__(self, model, agr_type="max", feat_dim=512, lifting_net=torch.nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        # Ensure LayerNorm uses the correct feature dimension
        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        else:
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):
        # Get pooled features and attention
        pooled_view, attention = self.aggregation_model(mvimages)
        
        # Ensure pooled_view is a 2D tensor of shape [batch_size, feat_dim]
        inter = self.inter(pooled_view)
        
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention