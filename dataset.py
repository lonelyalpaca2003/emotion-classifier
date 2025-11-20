from torch.utils.data import Dataset,transforms
import PIL 
import torch
from sklearn.preprocessing import LabelEncoder

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, encoder, transform=None, ):
        self.dataframe = dataframe
        self.encoder = encoder
        self.labels = torch.tensor(self.encoder.fit_transform(dataframe['label']))
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label