import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class MyCIFAR10(data.Dataset):
    def __init__(self):
        super(MyCIFAR10, self).__init__()
        self.data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    def __getitem__(self, item):
        return item, self.data.__getitem__(item)
    def __len__(self):
        return len(self.data)

def testdataset():
    return torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

def DataMeory(Dataset):
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    memory = dict([(i, []) for i in range(len(classes))])
    loader = data.DataLoader(Dataset, batch_size=1)
    for idx, img in loader:
        input, label = img
        memory[label.item()].append(idx.item())
    return memory




