from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from autoaugmentation import CIFAR10Policy
import collections

def unpickle(file):
    fo = open(file, 'rb').read()
    size = 64 * 64 * 3 + 1
    for i in range(50000):
        arr = np.fromstring(fo[i * size:(i + 1) * size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3, 64, 64)).transpose((1, 2, 0))
    return img, lab

class animal_dataset(Dataset): 
    def __init__(self, root, transform, mode, pred=[], probability=[], path=[], num_class=10):
        
        self.root = root
        self.transform = transform
        self.mode = mode
     
        self.train_dir = root + '/training/'
        self.test_dir = root + '/testing/'
        train_imgs = os.listdir(self.train_dir)
        test_imgs = os.listdir(self.test_dir)
        self.test_data = []
        self.test_labels = []
        noise_file1 = root +'/training_batch.json'
        noise_file2 = root +'/testing_batch.json'
        if mode == 'test':
            if os.path.exists(noise_file2):
                dict = json.load(open(noise_file2, "r"))
                self.test_data = dict['data']
                self.test_labels = dict['label']
            else:
                test_data = []
                test_labels = {}
                for img in test_imgs:
                    self.test_data.append(self.test_dir+img)
                    self.test_labels.append(int(img[0]))
                dicts = {}
                dicts['data'] = self.test_data
                dicts['label'] = self.test_labels
                json.dump(dicts, open(noise_file2, "w"))
        else:
            if os.path.exists(noise_file1):
                dict = json.load(open(noise_file1, "r"))
                train_data = dict['data']
                train_labels = dict['label']
            else:
                train_data = []
                train_labels = {}
                for img in train_imgs:
                    img_path = self.train_dir+img
                    train_data.append(img_path)
                    train_labels[img_path] = (int(img[0]))
                dicts = {}
                dicts['data'] = train_data
                dicts['label'] = train_labels
                json.dump(dicts, open(noise_file1, "w"))
            if self.mode == "all":
                self.train_data = train_data
                self.train_labels = train_labels
            elif self.mode == "labeled":
                pred_idx = pred.nonzero()[0]
                train_img = path
                self.train_data = [train_img[i] for i in pred_idx]
                self.probability = [probability[i] for i in pred_idx]
                self.train_labels = train_labels
                
                print("%s data has a size of %d" % (self.mode, len(self.train_data)))
            elif self.mode == "unlabeled":
                pred_idx = (1 - pred).nonzero()[0]
                train_img = path
                self.train_data = [train_img[i] for i in pred_idx]
                self.train_labels = train_labels
                print("%s data has a size of %d" % (self.mode, len(self.train_data)))

    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_data[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform[0](img)   
            img2 = self.transform[1](img)
            img3 = self.transform[2](img)   
            img4 = self.transform[3](img)
            return img1, img2, img3, img4, target, index, prob            
        elif self.mode=='unlabeled':
            img_path = self.train_data[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform[0](img)   
            img2 = self.transform[1](img)
            img3 = self.transform[2](img)   
            img4 = self.transform[3](img)
            return img1, img2, img3, img4
        elif self.mode=='all':
            img_path = self.train_data[index]
            target = self.train_labels[img_path]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)            
            return img, target, index, img_path        
        elif self.mode=='test':
            img_path = self.test_data[index]
            target = self.test_labels[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)
        
class animal_dataloader():  
    def __init__(self, root, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        
        transform_weak = transforms.Compose([
                    transforms.Resize(64),
                    transforms.RandomCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                ])
        
        transform_strong = transforms.Compose([
                    transforms.Resize(64),
                    transforms.RandomCrop(64),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                ])

        self.transform_train = {
                "warmup": transform_weak,
                "unlabeled": [transform_weak, transform_weak, transform_strong, transform_strong],
                "labeled": [transform_weak, transform_weak, transform_strong, transform_strong]
            }
        self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                ])    
    def run(self,mode,pred=[],prob=[],paths=[]):
        if mode=='warmup':
            warmup_dataset = animal_dataset(self.root, transform=self.transform_train["warmup"], mode='all')
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return warmup_loader
                                     
        elif mode == 'train':
            labeled_dataset = animal_dataset(self.root, transform=self.transform_train["labeled"], mode='labeled', pred=pred, probability=prob, path=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            unlabeled_dataset = animal_dataset(self.root, transform=self.transform_train["unlabeled"], mode='unlabeled', pred=pred, probability=prob, path=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_loader, unlabeled_loader
        elif mode == 'eval_train':
            eval_dataset = animal_dataset(self.root, transform=self.transform_test, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader
        elif mode == 'test':
            test_dataset = animal_dataset(self.root, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader