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

if __name__ == "__main__":
    device = reid.utilities.device()
    reid.utilities.manual_seed(777)
    # Prepare Datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_inv = torchvision.transforms.Compose([
        # torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    ])

    dataset = torchvision.datasets.MNIST('./data/mnist',
                                         transform=transform,
                                         download=True)
    dataset_train = reid.datasets.paired.DoublePairedVisionDataset(dataset)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   num_workers=0)
    # Create Model
    model = reid.models.simple_cnn()
    model.to(device)
    criterion = reid.losses.ContrastiveLoss(4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_sche = torch.optim.lr_scheduler.StepLR(optimizer,
                                              step_size=10,
                                              gamma=0.5)
    # Train
    loss_epoch = []
    num_epochs = 30
    for epoch in range(num_epochs):
        loss_batch = 0.0
        lr_sche.step()
        for i, batch in enumerate(dataloader_train, 0):
            inputs0, inputs1, labels = batch
            inputs0 = inputs0.to(device)
            inputs1 = inputs1.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = criterion(model(inputs0), model(inputs1), labels)

            loss.backward()
            optimizer.step()

            # print statistics
            loss_batch += loss
        loss_batch /= i + 1
        print(f'epoch: {epoch}, loss: {loss_batch}')
        loss_epoch.append(loss_batch)

        if epoch % 10 == 9:
            reid.utilities.save_model(model,
                                      f'contra_simple_mnist_{epoch + 1}')
