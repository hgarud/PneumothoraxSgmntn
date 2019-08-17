import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from data_loader3 import PnemoImgMask, Preprocess, Gray2RGB, ToTensor

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    phase="train"
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

            # Iterate over data.
        for i, data in enumerate(dataloaders):
            inputs = data['image'].to(device)
            target = data['target']
            target_id = target['image id'].to(device).squeeze(1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, target_id)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == target_id)

        epoch_loss = running_loss / 10712
        epoch_acc = running_corrects.double() / 10712

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model



if __name__ == "__main__":
    root = "/media/arshita/Windows/Users/arshi/Desktop/Project2"
    transformed_dataset = PnemoImgMask(root,
                transforms=transforms.Compose([Preprocess(),
                                                Gray2RGB(),
                                                ToTensor()]))
    print(len(transformed_dataset))
    dataloaders = torch.utils.data.DataLoader(transformed_dataset, batch_size=8,
                                         shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = [0,1]



    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                           num_epochs=25)
    ######################################################################################