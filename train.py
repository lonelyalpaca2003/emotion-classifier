from create_dataframe import create_dataframe
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config_parser import SimpleConfig
from dataloader import CustomImageDataloader
from dataset import CustomImageDataset
from training import Trainer
from model import ResnetModel
from early_stopping import EarlyStopping
from torchvision.models import resnet50

base_path = 'Data'
device = "cuda"
df = create_dataframe(base_path)

dataset = CustomImageDataset(df)
config = SimpleConfig('config.json')
train_dataloader = config.init_obj('dataloader', torch.utils.data, dataset)
val_dataloader = train_dataloader.validation_split()

resnet_model = ResnetModel(device = device, num_classes = 5)
model = resnet_model.create_initial_model()
criterion = config.init_obj['criterion', nn]

optimizer1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = config['phase1']['optimizer']['lr'], weight_decay = config['phase1']['optimizer']['weight_decay'])
scheduler1 = ReduceLROnPlateau(optimizer1, **config['phase1']['scheduler']['args'])
early_stopping = EarlyStopping(**config['phase1']['early_stopping']['args'])
trainer = Trainer(model, criterion, optimizer1, scheduler1, device = device, early_stopping = early_stopping)
print("Starting training phase 1")
trainer.fit(train_dataloader, val_dataloader, epochs = config['phase1']['epochs'])


### Phase 2 of fitting (we now unfreeze the final 2 layers to help our model finetune on our data)

for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer2 = Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = config['phase2']['optimizer']['lr']
                  , weight_decay = config['phase2']['optimizer']['weight_decay'])
scheduler2 = ReduceLROnPlateau(optimizer2, **config['phase2']['early_stopping']['args'])
early_stopping = EarlyStopping(**config['phase2']['early_stopping']['args'])
trainer = Trainer(model, criterion, optimizer2, scheduler2, device = device, early_stopping=early_stopping)
print("Starting training phase 2")
trainer.fit(train_dataloader, val_dataloader, epochs = config['phase2']['epochs'])






