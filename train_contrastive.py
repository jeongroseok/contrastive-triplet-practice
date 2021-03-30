import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.cuda
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms

import reid.datasets.paired
import reid.losses
import reid.models
import reid.utilities

# Get Device
device = reid.utilities.device()
reid.utilities.manual_seed(777)

# Prepare Datasets
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset_faces = torchvision.datasets.ImageFolder('./data/atnt-faces/train',
                                                 transform=transform)
dataset_train = reid.datasets.paired.DoublePairedVisionDataset(dataset_faces)
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=0)
# Create Model
model = reid.models.resnet18_custom(True)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
model.to(device)
criterion = reid.losses.ContrastiveLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Train
losses = []
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    lr_sche.step()
    for i, batch in enumerate(dataloader_train, 0):
        inputs0, inputs1, labels = batch
        inputs0 = inputs0.to(device)
        inputs1 = inputs1.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss: torch.Tensor = criterion(model(inputs0), model(inputs1), labels)
        loss.backward()
        optimizer.step()

        # print statistics

    print(f'epoch: {epoch}, loss: {loss}')
    losses.append(loss)

    if epoch % 10 == 9:
        torch.save(model.state_dict(), "./model_epoch_{}.pth".format(epoch))

plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), losses)
plt.show()

print('done')