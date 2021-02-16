from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class DefaultDataset(Dataset):
    def __init__(self, data_directory: str, csv_path: str, transforms):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
        self.data_directory = data_directory
        self.targets = list(self.df['class'])
        self.labels = {label: index for index, label in enumerate(list(set(self.targets)))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df['class'][idx]
        file = self.data_directory + '/' + str(label) + '/' + self.df['file'][idx]
        # return filename for visualizing
        return self.pil_loader(file), self.labels[label], file

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if self.transforms is not None:
                    return self.transforms(img.convert('RGB'))
                else:
                    return img.convert('RGB')