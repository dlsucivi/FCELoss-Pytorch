import os.path as osp
import torch
from torch.utils.data import Dataset
import cv2


class SD(Dataset):

    def __init__(self,
                 data_path,
                 ver,
                 split,
                 fold,
                 mode,
                 new_size,
                 image_transform=None):
        """Initializes the dataset

        Arguments:
            data_path {string} -- root path to the SD dataset
            ver {string} -- version of the SD dataset, i.e., SD-128, SD-198,
            SD-260
            split {string} -- dataset split, i.e., 5-5, 8-2
            fold {string} -- dataset fold for 8-2 split
            mode {string} -- current mode of the network
            new_size {int} -- rescaled size of the image
            image_transform {object} -- produces different dataset
            augmentation techniques
        """

        self.data_path = data_path
        self.ver = ver
        self.split = split
        self.fold = fold
        self.mode = mode
        self.new_size = new_size
        self.image_transform = image_transform

        self.ids = []
        self.targets = []

        if self.ver == 'SD-128':
            self.data_path = osp.join(self.data_path, 'SD-128')
        elif self.ver == 'SD-198':
            self.data_path = osp.join(self.data_path, 'SD-198')
        elif self.ver == 'SD-260':
            self.data_path = osp.join(self.data_path, 'SD-260')

        self.image_path = osp.join(self.data_path, 'images', '%s')

        if self.split == '5-5':
            annotation_path = osp.join(self.data_path, '5_5 Split')
        elif self.split == '8-2':
            annotation_path = osp.join(self.data_path, '8_2 Split')

        list_path = ''

        if self.mode == 'train':
           list_path = osp.join(annotation_path, 'train')
        elif self.mode == 'test':
           list_path = osp.join(annotation_path, 'val')

        if(self.fold):
            list_path += '_' + str(self.fold)
            
        list_path += '.txt'
        self.list_path = list_path

        print(f'Dataset: {list_path}')

        with open(list_path) as f:
            for line in f:
                values = line.split(' ')
                self.ids.append(values[0])
                self.targets.append(values[1].strip())

    def __len__(self):
        """Returns number of data in the dataset

        Returns:
            int -- number of data in the dataset
        """
        return len(self.ids)

    def __getitem__(self, index):
        """Returns an image and its corresponding class from the dataset

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor, string -- Tensor representation of the pulled image
            and string representation of the class of the image
        """
        image, target, _, _ = self.pull_item(index)

        return image, target

    def pull_item(self, index):
        """Returns an image, its corresponding class, height, and width from
        the dataset

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor, string, int, int -- Tensor representation of the
            pulled image, string representation of the class of the image,
            height of the image, width of the image
        """
        image_id = self.ids[index]

        image = cv2.imread(self.image_path % image_id)
        target = self.targets[index]
        height, width, _ = image.shape

        if self.image_transform is not None:
            image, target = self.image_transform(image, target)
            image = image[:, :, (2, 1, 0)]

        return torch.from_numpy(image).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        """Returns an image from the dataset represented as an ndarray

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            numpy.ndarray -- ndarray representation of the pulled image
        """

        image_id = self.ids[index]
        return cv2.imread(self.image_path % image_id, cv2.IMREAD_COLOR)

    def pull_target(self, index):
        """Returns a class corresponding to an image from the dataset

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            string -- string representation of the class of an image from the
            dataset
        """
        return self.targets[index]

    def pull_tensor(self, index):
        """Returns an image from the dataset represented as a tensor

        Arguments:
            index {int} -- index of the item to be pulled from the list of ids

        Returns:
            torch.Tensor -- Tensor representation of the pulled image
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
