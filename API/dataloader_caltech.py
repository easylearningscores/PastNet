
import os
import cv2
import numpy as np
import torch
import bisect
from torch.utils.data import Dataset

split_string = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"

def read_seq(path):
    f = open(path, 'rb+')
    string = f.read().decode('latin-1')
    str_list = string.split(split_string)
    # print(len(str_list))
    f.close()
    return str_list[1:]

def seq_to_images(bytes_string):
    res = split_string.encode('latin-1') + bytes_string.encode('latin-1')
    img = cv2.imdecode(np.frombuffer(res, np.uint8), cv2.IMREAD_COLOR)
    return img / 255.0

def load_caltech(root):
    file_list = [file for file in os.listdir(root) if file.split('.')[-1] == "seq"]
    print(file_list)
    for file in file_list[:1]:
        path = os.path.join(root, file)
        str_list, len = read_seq(path)
        imgs = np.zeros([len - 1, 480, 640, 3])
        idx = 0
        for str in str_list[1:]:
            imgs[idx] = seq_to_images(str)
            idx += 1
    return imgs

class Caltech(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, root, is_train=True, file_list=['V001.seq'], n_frames_input=4, n_frames_output=1):
        super().__init__()
        datasets = []
        for file in file_list:
            datasets.append(SingleCaltech(os.path.join(root, file), is_train=is_train,  n_frames_input=n_frames_input, n_frames_output=n_frames_output))
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets[:1])
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class SingleCaltech(Dataset):
    def __init__(self, root, is_train=True, n_frames_input=4, n_frames_output=1):
        super().__init__()
        self.root = root
        if is_train:
            self.length = 100
        else:
            self.length = 50

        self.input_length = n_frames_input
        self.output_length = n_frames_output

        self.sequence = None
        self.get_current_data()

        

    def get_current_data(self):
        str_list = read_seq(self.root)
        if self.length == 100:
            str_list = str_list[:104]
            self.sequence = np.zeros([104, 480, 640, 3])
        else:
            str_list = str_list[104:153]
            self.sequence = np.zeros([54, 480, 640, 3])
        
        for i, str in enumerate(str_list):
            self.sequence[i] = seq_to_images(str)

    def __getitem__(self, index):
        input = self.sequence[index: index + self.input_length]
        input = np.transpose(input, (0, 3, 1, 2))
        output = self.sequence[index + self.input_length: index + self.input_length + self.output_length]
        output = np.transpose(output, (0, 3, 1, 2))
        input = torch.from_numpy(input).contiguous().float()
        output = torch.from_numpy(output).contiguous().float()
        return input, output

    def __len__(self):
        return self.length


def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):
    
    file_list = [file for file in os.listdir(data_root) if file.split('.')[-1] == "seq"]
    # print(data_root)
    train_set = Caltech(root=data_root, is_train=True,
                            n_frames_input=4, n_frames_output=1, file_list=file_list)
    test_set = Caltech(root=data_root, is_train=False,
                           n_frames_input=4, n_frames_output=1, file_list=file_list)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std


if __name__ == "__main__":
    # data = load_caltech("/home/pan/workspace/simvp/SimVP-Simpler-yet-Better-Video-Prediction-master/data/caltech/USA/set01")
    file_list = [file for file in os.listdir("/home/pan/workspace/simvp/SimVP/data/caltech/USA/set01") if file.split('.')[-1] == "seq"]
    dataset = Caltech(root="/home/pan/workspace/simvp/SimVP/data/caltech/USA/set01", is_train=False, file_list=file_list)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True)
    
    for input, output in dataloader:
        print(input.shape, output.shape)