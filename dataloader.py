from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class CustomImageDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle : bool, validation_split : float, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.num_samples = len(dataset)

        self.sampler, self.valid_sampler = self.splitting(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(sampler = self.sampler, **self.init_kwargs)
    
    
    def splitting(self, split:float):
        if split == 0.0:
            return None, None

        assert split > 0, "Validation split must be a valid value"
        len_valid = int(self.num_samples * split)
        idx = np.arange(self.num_samples)
        np.random.shuffle(idx)

        valid_idx = idx[0:len_valid]
        train_idx = idx[len_valid:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.shuffle = False
        self.num_samples = len(train_idx)

        return train_sampler, valid_sampler
    
    def validation_split(self):
        if self.valid_sampler is not None:
            return DataLoader(sampler = self.valid_sampler, **self.init_kwargs)
        else: 
            return None



        

