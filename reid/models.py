import torch
import torch.nn
import torchvision.models


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(torch.nn.Linear(128,
                                                      512), torch.nn.ReLU(),
                                      torch.nn.Linear(512, 128),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def simple_cnn():
    return SimpleCNN()


def resnet18_custom(pretrained: bool = True, progress=True, **kwargs):
    model = torchvision.models.resnet18(pretrained, progress, **kwargs)

    model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features,
                                                   2048), torch.nn.ReLU(),
                                   torch.nn.Linear(2048, 1024),
                                   torch.nn.ReLU(), torch.nn.Linear(1024, 128))
    return model


def resnet50_custom(pretrained: bool = True, progress=True, **kwargs):
    model = torchvision.models.resnet50(pretrained, progress, **kwargs)

    model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features,
                                                   2048), torch.nn.ReLU(),
                                   torch.nn.Linear(2048, 1024),
                                   torch.nn.Linear(1024, 1024),
                                   torch.nn.ReLU(), torch.nn.Linear(1024, 256))
    return model