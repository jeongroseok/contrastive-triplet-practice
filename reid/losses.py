import torch
import torch.nn
import torch.nn.functional


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, outputs1, outputs2, labels):
        distances = torch.nn.functional.pairwise_distance(outputs1,
                                                          outputs2,
                                                          keepdim=True)
        losses = torch.mean(
            ((1 - labels) * (distances**2)) +  # label이 0일때, 거리가 클수록
            (labels * (torch.clamp(self.margin - distances, min=0.0)**2))
        )  # label이 1일때, 거리가 가까울수록
        return losses
