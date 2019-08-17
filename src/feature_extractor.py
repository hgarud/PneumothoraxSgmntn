import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from transform import PnemoImgMask, Preprocess, Gray2RGB, ToTensor


class ResNet18Top(nn.Module):
    def __init__(self, original_model):
        super(ResNet18Top, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.features(x)
        return x

def extract_features(dataloaders):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet18_model = models.resnet18(pretrained=True)
    model = ResNet18Top(resnet18_model)
    model = model.to(device)
    model.eval()
    for i, data in enumerate(dataloaders):
        inputs = data['image'].to(device)
        outputs = model(inputs)
        print("Features Extracted: ", outputs.shape)


if __name__ == "__main__":
    transformed_dataset = PnemoImgMask(transforms=transforms.Compose([Preprocess(),
                                                Gray2RGB(),
                                                ToTensor()]))
    print(len(transformed_dataset))
    dataloaders = torch.utils.data.DataLoader(transformed_dataset, batch_size=8,
                                         shuffle=True, num_workers=4)

    extract_features(dataloaders)
