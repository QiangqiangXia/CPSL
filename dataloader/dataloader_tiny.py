from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

# from vision import VisionDataset

from PIL import Image
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from autoaugment_tiny import CIFAR10Policy, ImageNetPolicy
from tiny_pairflip_noise import *

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


transform_none_100_compose = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform_weak_100_compose = transforms.Compose(
    [
        transforms.RandomCrop(64),
        # transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.4, hue=0.07),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform_strong_100_compose = transforms.Compose(
    [
        transforms.RandomCrop(64),
        transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.4, hue=0.07),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)
    clsa, class_to_idx = find_classes(directory)
    # print(clsa,class_to_idx)
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances,class_to_idx

class tiny_imagenet_dataset(Dataset):
    def __init__(self, root, transform, mode, ratio, noise_mode, noise_file = '', num_samples=10000, pred=[], probability=[], paths=[], num_class=200):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.ratio = ratio
        self.noise_mode = noise_mode

        ### Get the instances and check if it is right
        data_folder = '../tiny/train/'
        train_instances, dict_classes = make_dataset(data_folder, extensions = IMG_EXTENSIONS)

        ## Validation Files
        data_folder = '../tiny/val/'
        #val_instances = make_dataset(data_folder, extensions = IMG_EXTENSIONS)
        val_text = '../tiny/val/val_annotations.txt'
        val_img_files = '../tiny/val/images'
        num_class  = 200 
        num_sample = 100000
        #data_folder = '../tiny/test/'
        #test_instances = make_dataset(data_folder, extensions = IMG_EXTENSIONS)

        ## Load these instances->(data, label) into custom dataloader    
        self.val_labels   = {}
        self.train_labels = []
        
        self.train_images = []
        self.val_imgs 	= []


        for kk in range(len(train_instances)):
            path_ind, label = list(train_instances[kk])
            self.train_labels.append(label)
            self.train_images.append(path_ind)

        if os.path.exists(noise_file):
            noise_label = json.load(open(noise_file,"r"))
        else:
            noise_label = []
            idx = list(range(num_sample))
            random.shuffle(idx)
            num_noise = int(ratio*len(idx))
            noise_idx = idx[:num_noise]

            ## Check the Noise Type
            if noise_mode=="asym":
                noiselabel, noise_rate = noisify('tiny_imagenet', num_class, np.array(train_label), 'pairflip', ratio, 0)
                num = 0
                for kk in self.train_images:
                    noise_label[kk] = noiselabel[num]
                    num += 1
            else:
                for i in range(num_sample):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            noiselabel = random.randint(0,num_class-1)
                            noise_label.append(noiselabel)

                        elif noise_mode == 'pair_flip':
                            noiselabel = self.pair_flipping[train_label[i]]
                            noise_label.append(noiselabel) 
                    else:
                        noise_label.append(self.train_labels[i])
            
                print("Save noisy labels to %s ..."%noise_file)
                json.dump(noise_label,open(noise_file,"w")) 

        if self.mode == 'all':
            self.train_labels = noise_label
            self.train_imgs = self.train_images
            print("Number of Samples:", len(self.train_imgs))

        elif self.mode == "labeled":
            pred_idx = pred.nonzero()[0]
            self.probability = [probability[i] for i in pred_idx]
            self.train_imgs = [self.train_images[i] for i in pred_idx]
            self.train_labels = [noise_label[i] for i in pred_idx]    
            print("%s data has a size of %d"%(self.mode, len(self.train_imgs)))                

        elif self.mode == "unlabeled":
            pred_idx = (1-pred).nonzero()[0]  
            self.train_imgs = [self.train_images[i] for i in pred_idx]
            self.train_labels = [noise_label[i] for i in pred_idx]   
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

        elif self.mode == 'val':
            with open(val_text,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = '%s/'%val_img_files+entry[0]
                    self.val_labels[img_path] = int(dict_classes[entry[1]])
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB') 

            ## Weakly and Strongly Augmeneted Copies 
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4, target, index, prob 

        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')

            ## Weakly and Strongly Augmeneted Copies 
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4

        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index

        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)

class tinyImagenet_dataloader():  
    def __init__(self, root, batch_size, num_workers, ratio,  noise_mode, noise_file):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.ratio = ratio
        self.noise_mode = noise_mode
        self.noise_file = noise_file

        self.transforms = {
            "warmup": transform_weak_100_compose,
            "unlabeled": [
                        transform_weak_100_compose,
                        transform_weak_100_compose,
                        transform_strong_100_compose,
                        transform_strong_100_compose
                    ],
            "labeled": [
                        transform_weak_100_compose,
                        transform_weak_100_compose,
                        transform_strong_100_compose,
                        transform_strong_100_compose
                    ],
            "test": None,
        }        

        self.transform_test = transforms.Compose([
				transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])        

    def run(self,mode,pred=[],prob=[]):        
        if mode=='warmup':
            warmup_dataset = tiny_imagenet_dataset(self.root,transform=self.transforms["warmup"], mode='all', ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=int(self.batch_size*4),
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader

        elif mode=='train':
            labeled_dataset = tiny_imagenet_dataset(self.root,transform=self.transforms["labeled"], mode='labeled',  ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file, pred=pred, probability=prob)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True, drop_last= True,
                num_workers=self.num_workers)

            unlabeled_dataset = tiny_imagenet_dataset(self.root,transform=self.transforms["unlabeled"], mode='unlabeled',  ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file, pred=pred, probability=prob)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True, drop_last= True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader

        elif mode=='eval_train':
            eval_dataset = tiny_imagenet_dataset(self.root,transform=self.transform_test, mode='all', ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=250,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader                    

        elif mode=='val':
            val_dataset = tiny_imagenet_dataset(self.root,transform=self.transform_test, mode='val', ratio = self.ratio, noise_mode = self.noise_mode, noise_file=self.noise_file)
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=250,
                shuffle=False,
                num_workers=self.num_workers)             
            return val_loader             
