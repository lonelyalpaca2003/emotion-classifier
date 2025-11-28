from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights
import PIL 
import torch
from sklearn.preprocessing import LabelEncoder

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, encoder):
        self.dataframe = dataframe
        self.encoder = encoder
        self.encoder.fit(dataframe['label'])
        self.labels = torch.tensor(self.encoder.fit_transform(dataframe['label']))
        self.transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
        

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label