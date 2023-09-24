import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iKohYoung(iData):
    use_path = False
    train_trsf = [
    ]
    test_trsf = [
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    class_order = (np.arange(1000)+1).tolist()

    def download_data(self):
        import os
        from skimage import io
        import cv2

        data_dir = "../data"
        save_train_dir = os.path.join(data_dir, 'train_data')
        save_valid_dir = os.path.join(data_dir, 'valid_data')

        x_train_path = os.path.join(save_train_dir, 'x_data.npy')
        y_train_path = os.path.join(save_train_dir, 'y_data.npy')
        x_valid_path = os.path.join(save_valid_dir, 'x_data.npy')
        y_valid_path = os.path.join(save_valid_dir, 'y_data.npy')

        is_x_train = os.path.isfile(x_train_path)
        is_y_train = os.path.isfile(y_train_path)
        is_x_valid = os.path.isfile(x_valid_path)
        is_y_valid = os.path.isfile(y_valid_path)

        if all([is_x_train, is_y_train, is_x_valid, is_y_valid]):
            x_train = np.load(x_train_path, allow_pickle=True)
            y_train = np.load(y_train_path, allow_pickle=True)
            x_valid = np.load(x_valid_path, allow_pickle=True)
            y_valid = np.load(y_valid_path, allow_pickle=True)

        else:
            train_dir = os.path.join(data_dir, 'train/')
            validation_num = 20

            # Divide data 0-129 for training, 130-150 for validation.
            x_train = []
            x_valid = []
            y_train = []
            y_valid = []

            # Split input data.
            for folder_idx in range(1, 1001):
                for img_idx in range(0, 150):
                    path = os.path.join(train_dir, str(folder_idx))
                    path = path + '/' + str(img_idx)+'.png'
                    img = io.imread(path)
                    img = cv2.resize(img, (128, 128))
                    if img_idx < 150-validation_num:
                        x_train.append(img)
                        y_train.append(np.array([folder_idx]))
                    else:
                        x_valid.append(img)
                        y_valid.append(np.array([folder_idx]))

            # Convert list to numpy
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_valid = np.array(x_valid)
            y_valid = np.array(y_valid)

            os.makedirs(save_train_dir, exist_ok=True)
            os.makedirs(save_valid_dir, exist_ok=True)

            np.save(x_train_path, x_train)
            np.save(y_train_path, y_train)
            np.save(x_valid_path, x_valid)
            np.save(y_valid_path, y_valid)

        self.train_data, self.train_targets = x_train, y_train
        self.test_data, self.test_targets = x_valid, y_valid
