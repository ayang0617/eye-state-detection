import readtxt
import torch
from torchvision import transforms
import cv2

from torch.utils.data.dataset import Dataset

class CustomDateset(Dataset):
    def __init__(self, location,transforms=None):
        self.txt = readtxt.ReadTxt(location)
        self.transforms = transforms

    def __getitem__(self, index):

        path = self.txt[index][0]
        n = self.txt[index][1]
        n = int(n)
        label_tensor = torch.Tensor([n])

        img = cv2.imread(path, 1)
        #cv2.resize(src=img, (50, 50), dst=img)
        img = cv2.resize(img, (50, 50))
        #image_to_tensor = transforms.Compose([transforms.ToTensor()])
        image_to_tensor =transforms.ToTensor()
        img_tensor = image_to_tensor(img)
        return img_tensor, label_tensor



    def __len__(self):
        return len(self.txt)

if __name__ == '__main__':
    location = "/Users/ayang/PycharmProjects/pythonProject/train.txt"
    customdataset = CustomDateset(location)
    for index, (img_tensor, label_tensor) in enumerate(customdataset):
        #print(index)
        print(img_tensor)
        print(label_tensor)