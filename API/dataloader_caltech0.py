
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

split_string = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"

def read_seq(path):
    f = open(path, 'rb+')
    string = f.read().decode('latin-1')
    str_list = string.split(split_string)
    print(len(str_list))
    f.close()
    return str_list, len(str_list)

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
    return imgs.transpose(0, 3, 1, 2)

class Caltech(Dataset):
    def __init__(self, root, is_train=True, n_frames_input=4, n_frames_output=1):
        super().__init__()
        self.root = root
        print("loading .seq file list")
        self.file_list = [file for file in os.listdir(self.root) if file.split('.')[-1] == "seq"]
        
        if is_train:
            self.file_list = self.file_list[:-1]
        else:
            self.file_list = self.file_list[-1:]

        print("loading file list done, file list: ", self.file_list)
        self.length = 0

        self.input_length = n_frames_input
        self.output_length = n_frames_output

        self.current_seq = None
        self.current_length = 0
        self.current_file_index = 0

        self.get_next = True 

        self.get_total_len(root)
        self.get_current_data()


    def get_total_len(self, root):
        print("calculating total length")
        count = 0
        for file in self.file_list:
            path = os.path.join(root, file)
            _, len = read_seq(path)
            count += (len - 5)
        self.length = count
        print("calculating total length done, total length: ", self.length)

    def get_current_data(self):
        print("getting current sequence")
        if self.current_file_index >= len(self.file_list):
            self.get_next = False
            return
        current_file = os.path.join(self.root, self.file_list[self.current_file_index])
        str_list, length = read_seq(current_file)
        self.current_length = length - 5
        self.current_seq = np.zeros([length - 1, 480, 640, 3])
        for i, str in enumerate(str_list[1:]):
            self.current_seq[i] = seq_to_images(str)
        print("getting current sequence done, the shape:", self.current_seq.shape)

    def get_next_seq(self):
        print("getting next sequence")
        self.current_file_index += 1
        self.get_current_data()
        self.get_next = False

    def __getitem__(self, index):
        if index >= self.current_length:
            self.get_next = True
        if self.get_next:
            self.get_next_seq()
        input = self.current_seq[index: index + self.input_length]
        output = self.current_seq[index + self.input_length: index + self.input_length + self.output_length]
        input = torch.from_numpy(input).contiguous().float()
        output = torch.from_numpy(output).contiguous().float()
        return input, output

    def __len__(self):
        return self.current_length


def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    train_set = Caltech(root=data_root, is_train=True,
                            n_frames_input=10, n_frames_output=10, num_objects=[2])
    test_set = Caltech(root=data_root, is_train=False,
                           n_frames_input=10, n_frames_output=10, num_objects=[2])

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
    dataset = Caltech(root="/home/pan/workspace/simvp/SimVP-Simpler-yet-Better-Video-Prediction-master/data/caltech/USA/set01")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, drop_last=True)
    
    for input, output in dataloader:
        print(input.shape, output.shape)