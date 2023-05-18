import numpy as np
import torch
import torch.utils.data as data


class MovingMnistSequence(data.Dataset):
    def __init__(self, train=True, shuffle=True, root='./data', transform=None):
        super().__init__()
        if train:
            npz = 'mnist_train.npz'
            self.data = np.load(f'{root}/{npz}')['input_raw_data']
        else:
            npz = 'mnist_train.npz'
            self.data = np.load(f'{root}/{npz}')['input_raw_data'][:10000]


        self.transform = transform
        self.data = self.data.transpose(0, 2, 3, 1)

    def __len__(self):
        return self.data.shape[0] // 20

    def __getitem__(self, index):
        imgs = self.data[index * 20: (index + 1) * 20]
        imgs_tensor = torch.zeros([20, 1, 64, 64])
        if self.transform is not None:
            for i in range(imgs.shape[0]):
                imgs_tensor[i] = self.transform(imgs[i])
        return imgs_tensor


def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    train_set = MovingMnistSequence(root=data_root, is_train=True,
                            n_frames_input=10, n_frames_output=10, num_objects=[2])
    test_set = MovingMnistSequence(root=data_root, is_train=False,
                           n_frames_input=10, n_frames_output=10, num_objects=[2])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std
