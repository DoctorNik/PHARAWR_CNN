import torch
import torch.nn as nn
from torchvision import models
from torch import optim

class ActionClassifier(nn.Module):
    def __init__(self, ntargets):
        super().__init__()
        resnet = models.resnet50(pretrained=True, progress=True)
        modules = list(resnet.children())[:-1]  # удаляем последний слой
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(resnet.fc.in_features),
            nn.Dropout(0.2),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, ntargets)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

# загрузки модели с сохранёнными весами
def load_model(weights_path, ntargets, device):
    model = ActionClassifier(ntargets).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model
