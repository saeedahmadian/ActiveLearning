import torch
import os
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from Datasets import MyCIFAR10, testdataset, DataMeory
from Model import Net
from CSmapler import ALSampler
import math

TrainDataset = MyCIFAR10()
if 'Memory.csv' not in os.listdir('data'):
    print('It takes a while to save meta data ..\nplease wait for 2 minutes...')
    DataMemory = DataMeory(TrainDataset)
    pd.DataFrame(DataMemory).to_csv(os.path.join('data', 'Memory.csv'))
else:
    DataMemory = pd.read_csv(os.path.join('data', 'Memory.csv'), index_col=0).to_dict(orient='list')
    DataMemory = {int(k): v for k, v in DataMemory.items()}


class UncertaintyPoolAL(object):
    def __init__(self, total_ratio=.05, num_classes=10,
                 pooling_ratio=.5, sampling_limit=10, epochs=2, batch_size=4,lr=.005):
        """
               :param total_ratio: Total ratio of training_size which is 50000 images
               It is considered in case we want to shrink the main training set samples
               :param num_classes: number of labels
               :param pooling_ratio: the ratio of pooling size for or better to say the
               ratio of trainig set that we can ask for label
               :param sampling_limit: numer of permitted sampling
               :param epochs: number of epoch for each time training the network
               :param batch_size: numebr of batch size for training
               :param lr: learning rate
        """
        self.loss = nn.CrossEntropyLoss()
        self.UnLabeledData = TrainDataset
        self.TestData = testdataset()
        self.total_ratio = total_ratio
        self.num_classes = num_classes
        self.sampler = ALSampler
        self.pool_ratio = pooling_ratio
        self.sampling_limit = sampling_limit
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr= lr
        self.pooling_methods = {'least-confidence': self.least_confidence,
                                'ratio-confidence': self.ratio_confidence,
                                'entropy': self.entropy}

    def Initidx(self):
        """
            :return: len of pool size and index required of train dataset
        """
        n_per_class = int((len(TrainDataset) * self.total_ratio) / self.num_classes)
        self.pool_index = []
        for val in DataMemory.values():
            self.pool_index += np.random.choice(val, n_per_class, replace=False).tolist()
        np.random.shuffle(self.pool_index)
        self.pool_index = set(self.pool_index)

    def train(self, indices):
        """
        :param indices: indices of data that needs to be trained
        :return: train the network
        """
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            indices_list = list(indices)
            np.random.shuffle(indices_list)
            sampler = self.sampler(indices_list)
            trainloader = data.DataLoader(self.UnLabeledData, batch_size=self.batch_size,
                                          shuffle=False, sampler=sampler)
            running_loss = 0.0
            for i, (idx, img) in enumerate(trainloader):
                # get the inputs
                inputs, labels = img

                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                obj = self.loss(outputs, labels)
                obj.backward()
                self.optim.step()

                # print statistics
                running_loss += obj.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def test(self):
        """
        :return: accuracy of trained network
        """
        correct = 0
        total = 0
        batch_size = int(len(self.TestData) / 10)
        with torch.no_grad():
            for img in data.DataLoader(self.TestData, batch_size=batch_size):
                images, labels = img
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                correct / total * 100.))
        return correct / total * 100.

    def Run_experiment(self, method, stopping_accuracy=.95):
        """
        :param method: the method of querying that can be 'least-confidence',
        'ratio-confidence','entropy'
        :param stopping_accuracy: stopping criteria for sampling method
        :return: list of accuracy per iteration, indices of queried samples
        """
        self.net = Net()
        # self.optim = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)
        self.to_be_labeled = set()
        if self.total_ratio == 1:
            self.pool_index = set(range(0, len(TrainDataset)))
            self.pool_len = len(self.pool_index)
        else:
            self.Initidx()
            self.pool_len = len(self.pool_index)
        sample_chunk = int((self.pool_len * self.pool_ratio) / self.sampling_limit)
        self.to_be_labeled = set(np.random.choice(list(self.pool_index), sample_chunk, replace=False).tolist())
        self.pool_index -= self.to_be_labeled
        # subsampler = self.sampler(list(self.to_be_labeled))
        limit = self.pool_len * self.pool_ratio
        self.accuracy = []
        self.accuracy.insert(0, 0)
        strategy = self.pooling_methods[method]
        print('{} Querying process begins ...'.format(method))
        cc = 1
        while (len(self.to_be_labeled) < limit or self.accuracy[-1] < stopping_accuracy):
            print('iteration {} out of {}'.format({cc}, {self.sampling_limit}))
            self.train(self.to_be_labeled)
            newidx = strategy(self.sampler(list(self.pool_index)), sample_chunk)
            self.pool_index -= set(newidx)
            self.to_be_labeled.update(set(newidx))
            self.accuracy.append(self.test())
            cc += 1
        return self.accuracy

    def bench_mark(self):
        """
        :return: train thee model using random sampling
        """
        if self.total_ratio == 1:
            self.pool_index = set(range(0, len(TrainDataset)))
            self.pool_len = len(self.pool_index)
        else:
            self.Initidx()
            self.pool_len = len(self.pool_index)
        limit = int(self.pool_len * self.pool_ratio)
        indecis_list = np.random.choice(list(self.pool_index), limit, replace=False).tolist()
        Testnet = Net()
        # Testoptim = optim.SGD(Testnet.parameters(), lr=self.lr, momentum=0.9)
        Testoptim = optim.Adam(Testnet.parameters(), lr=self.lr)
        Testloss = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            np.random.shuffle(indecis_list)
            sampler = self.sampler(indecis_list)
            trainloader = data.DataLoader(self.UnLabeledData, batch_size=self.batch_size,
                                          shuffle=False, sampler=sampler)
            running_loss = 0.0
            for i, (idx, img) in enumerate(trainloader):
                # get the inputs
                inputs, labels = img

                Testoptim.zero_grad()

                outputs = Testnet(inputs)
                obj = Testloss(outputs, labels)
                obj.backward()
                Testoptim.step()

                running_loss += obj.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        correct = 0
        total = 0
        batch_size = int(len(self.TestData) / 10)
        with torch.no_grad():
            for img in data.DataLoader(self.TestData, batch_size=batch_size):
                images, labels = img
                outputs = Testnet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the Bench mark on the 10000 test images: %d %%' % (
                correct / total * 100.))
        return correct / total * 100.

    def least_confidence(self, sampler, first_k):
        """
        :param sampler: class of sampler that choose from pool data base
        :param first_k: first k number of best samples
        :return: indices of choosen samples based on least_confidence
        """
        batch_size = int(len(self.pool_index) / first_k)
        loader = data.DataLoader(self.UnLabeledData, batch_size=batch_size, sampler=sampler)
        confidence_list = []
        with torch.no_grad():
            for idx, sample in loader:
                img, label = sample
                pred = F.softmax(self.net(img), dim=-1)
                val, _ = torch.max(pred, 1)
                max_val, max_ind = torch.topk(1 - val, 1)
                idx_max = idx[max_ind.item()].item()
                confidence_list.append(idx_max)

        return confidence_list

    def ratio_confidence(self, sampler, first_k):
        """
          :param sampler: class of sampler that choose from pool data base
          :param first_k: first k number of best samples
          :return: indices of choosen samples based on ratio confidence
          """
        batch_size = int(len(self.pool_index) / first_k)
        loader = data.DataLoader(self.UnLabeledData, batch_size=batch_size, sampler=sampler)
        confidence_list = []
        with torch.no_grad():
            for idx, sample in loader:
                img, label = sample
                pred = F.softmax(self.net(img), dim=-1)
                val, _ = torch.sort(pred, 1, descending=True)
                ratio = (val.data[:, 1] / (val.data[:, 0] + 1e-6))
                max_val, max_ind = torch.topk(ratio, 1, largest=True)
                idx_max = idx[max_ind.item()].item()
                confidence_list.append(idx_max)
        return confidence_list

    def entropy(self, sampler, first_k):
        """
          :param sampler: class of sampler that choose from pool data base
          :param first_k: first k number of best samples
          :return: indices of choosen samples based on entropy
          """
        batch_size = int(len(self.pool_index) / first_k)
        loader = data.DataLoader(self.UnLabeledData, batch_size=batch_size, sampler=sampler)
        confidence_list = []
        with torch.no_grad():
            for idx, sample in loader:
                img, label = sample
                pred = F.softmax(self.net(img), dim=-1)
                log_probs = pred * torch.log2(pred)
                raw_entropy = 0 - torch.sum(log_probs, 1)
                entropy = raw_entropy / math.log2(self.num_classes)
                max_val, max_ind = torch.topk(entropy, 1)
                idx_max = idx[max_ind.item()].item()
                confidence_list.append(idx_max)
        return confidence_list

