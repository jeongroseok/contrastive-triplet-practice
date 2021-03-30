from typing import Tuple
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.utils
import torchvision.transforms
import random
import matplotlib.pyplot as plt


class DoublePairedVisionDataset(
        torch.utils.data.dataset.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset: torchvision.datasets.VisionDataset):
        self._dataset = dataset

        # 라벨 인덱스별로 나뉜 샘플 인덱스
        groupped_by_label = {}
        for index_sample, sample in enumerate(dataset):
            label = sample[1]

            # label key가 없으면 생성
            if label not in groupped_by_label:
                groupped_by_label[label] = ()

            groupped_by_label[label] += (index_sample, )

        self._indexes_groupped = groupped_by_label
        self._classes = sorted(list(self._indexes_groupped.keys()))

    def __getitem__(self,
                    index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        anchor: Tuple[torch.Tensor, int] = self._dataset[index]

        label_anchor = anchor[1]

        label_other = label_anchor
        if random.randint(0, 1):  # 0 or 1
            label_other = random.choice(
                list(range(label_anchor)) +
                list(range(label_anchor + 1, len(self._classes))))

        other = self.__get_random_item_by_label_index(label_other)
        return anchor[0], other[0], int(label_anchor != label_other)

    def __get_random_item_by_label_index(self, label_index: int
                                         ) -> Tuple[torch.Tensor, int]:
        indexes = self._indexes_groupped[label_index]
        index = indexes[random.randrange(0, len(indexes))]
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)


class TriplePairedVisionDataset(
        torch.utils.data.dataset.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset: torchvision.datasets.VisionDataset):
        self._dataset = dataset

        # 라벨 인덱스별로 나뉜 샘플 인덱스
        groupped_by_label = {}
        for index_sample, sample in enumerate(dataset):
            label = sample[1]

            # label key가 없으면 생성
            if label not in groupped_by_label:
                groupped_by_label[label] = ()

            groupped_by_label[label] += (index_sample, )

        self._indexes_groupped = groupped_by_label
        self._classes = sorted(list(self._indexes_groupped.keys()))

    def __getitem__(self, index: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor: Tuple[torch.Tensor, int] = self._dataset[index]

        label_anchor = anchor[1]
        label_positive = label_anchor
        label_negative = random.choice(
            list(range(label_anchor)) +
            list(range(label_anchor + 1, len(self._classes))))

        positive = self.__get_random_item_by_label_index(label_positive)
        nagative = self.__get_random_item_by_label_index(label_negative)
        return anchor[0], positive[0], nagative[0]

    def __get_random_item_by_label_index(self, label_index: int
                                         ) -> Tuple[torch.Tensor, int]:
        indexes = self._indexes_groupped[label_index]
        index = indexes[random.randrange(0, len(indexes))]
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)


def test():
    def show_double_pair(pair: Tuple[torch.Tensor, torch.Tensor, int]):
        item_0, item_1, label = pair
        items = torch.cat((item_0.unsqueeze(dim=0), item_1.unsqueeze(dim=0)))
        items = torchvision.utils.make_grid(items).permute(1, 2, 0)
        print(label)
        plt.imshow(items)
        plt.show()

    def show_triple_pair(
            pair: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        item_0, item_1, item_2 = pair
        items = torch.cat((item_0.unsqueeze(dim=0), item_1.unsqueeze(dim=0),
                           item_2.unsqueeze(dim=0)))
        items = torchvision.utils.make_grid(items).permute(1, 2, 0)
        plt.imshow(items)
        plt.show()

    mnist = torchvision.datasets.MNIST(
        './data/mnist',
        download=True,
        transform=torchvision.transforms.ToTensor())

    double_pair = DoublePairedVisionDataset(mnist)
    show_double_pair(double_pair[0])
    show_double_pair(double_pair[1])
    show_double_pair(double_pair[2])
    show_double_pair(double_pair[3])

    triple_pair = TriplePairedVisionDataset(mnist)
    show_triple_pair(triple_pair[0])
    show_triple_pair(triple_pair[1])
    show_triple_pair(triple_pair[2])
    show_triple_pair(triple_pair[3])
    print('done')
