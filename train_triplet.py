from typing import Tuple
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
import visdom

import reid.datasets.triplet
import reid.losses
import reid.models
import reid.utilities

if __name__ == "__main__":
    # Visualization
    vis = visdom.Visdom()
    vis.close(env="main")
    loss_plt = vis.line(Y=torch.Tensor(1).zero_(),
                        opts=dict(title='loss',
                                  legend=['loss'],
                                  showlegend=True))
    dist0_plt = vis.line(Y=torch.Tensor(1).zero_(),
                         opts=dict(title='dist0',
                                   legend=['dist'],
                                   showlegend=True))
    dist1_plt = vis.line(Y=torch.Tensor(1).zero_(),
                         opts=dict(title='dist1',
                                   legend=['dist'],
                                   showlegend=True))

    def value_tracker(value_plot, value, num):
        '''num, loss_value, are Tensor'''
        vis.line(X=num, Y=value, win=value_plot, update='append')

    # Get Device
    device = reid.utilities.device()
    reid.utilities.manual_seed(0)

    # Create Datasets & Dataloaders
    DATASET_ROOT = './data/prid_2011'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset_train = reid.datasets.triplet.TripletImageFolder(
        root=f'{DATASET_ROOT}/train', transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=16,
                                                   shuffle=True,
                                                   num_workers=0)

    # Create Model
    model = reid.models.siamese_resnet18(True)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True
    model.to(device)

    criterion = torch.nn.TripletMarginLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_sche = torch.optim.lr_scheduler.StepLR(optimizer,
                                              step_size=10,
                                              gamma=0.5)

    # Train
    for epoch in range(100):
        running_loss = 0.0
        lr_sche.step()
        for i, batch in enumerate(dataloader_train, 0):
            inputs_anchor, inputs_positive, inputs_negative = batch
            inputs_anchor = inputs_anchor.to(device)
            inputs_positive = inputs_positive.to(device)
            inputs_negative = inputs_negative.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs_anchor = model(inputs_anchor)
            outputs_positive = model(inputs_positive)
            outputs_negative = model(inputs_negative)
            loss: torch.Tensor = criterion(outputs_anchor, outputs_positive,
                                           outputs_negative)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 3:  # print every 10 mini-batches
                value_tracker(
                    loss_plt, torch.Tensor([running_loss / 10]),
                    torch.Tensor([i + epoch * len(dataloader_train)]))
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        if epoch % 10 == 9:
            torch.save(model.state_dict(),
                       "./triplet_epoch_{}.pth".format(epoch))

    print('Finished Training')
