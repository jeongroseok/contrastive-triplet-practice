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

import reid.datasets.triplet
import reid.losses
import reid.models
import reid.utilities


def visualize_batch(model,
                    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    transform):
    anchors, positives, negatives = batch

    for i in range(len(anchors)):
        anchor = anchors[i].unsqueeze(dim=0).to(device)
        positive = positives[i].unsqueeze(dim=0).to(device)
        negative = negatives[i].unsqueeze(dim=0).to(device)

        # calc distance between anchor and X
        output_anchor = model(anchor)
        output_positive = model(positive)
        output_negative = model(negative)
        distance_positive = torch.nn.functional.pairwise_distance(
            output_anchor, output_positive)
        distance_negative = torch.nn.functional.pairwise_distance(
            output_anchor, output_negative)

        # make grid
        imgs = torch.cat(
            (transform(anchor), transform(positive), transform(negative)), 0)
        plt.imshow(torchvision.utils.make_grid(imgs).cpu().permute(1, 2, 0))
        plt.text(75,
                 8,
                 'p: {:.2f}, n: {:.2f}'.format(distance_positive.item(),
                                               distance_negative.item()),
                 fontweight='bold',
                 bbox={
                     'facecolor': 'white',
                     'alpha': 0.8,
                     'pad': 10
                 })
        plt.show()


if __name__ == "__main__":
    # prepare
    device = 'cpu'

    DATASET_ROOT = './data/prid_2011'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_inv = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    ])
    dataset_test = reid.datasets.triplet.TripletImageFolder(
        root=f'{DATASET_ROOT}/test', transform=transform)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=0)

    # create model
    model = reid.models.siamese_resnet18(False)
    model.load_state_dict(torch.load('./triplet_epoch_29.pth'))
    model.to(device).eval()

    # visualize
    batch = next(iter(dataloader_test))
    visualize_batch(model, batch, transform_inv)
    print('done')
