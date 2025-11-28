from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from config_parser import SimpleConfig
from early_stopping import EarlyStopping

class ResnetModel:
    def __init__(self, device, num_classes = 5):
        self.device = device
        self.num_classes = num_classes
    def create_initial_model(self):
        config = SimpleConfig('config.json')
        model_config = config['model']
        if model_config['type'] == 'resnet50':
            model = resnet50(weights='IMAGENET1K_V1')
            model.fc = nn.Sequential(
            nn.Linear(2048, 512),           
            nn.BatchNorm1d(512),            
            nn.ReLU(),
            nn.Dropout(0.5),                
            nn.Linear(512, self.num_classes)
        )
            for param in model.parameters():
                param.requires_grad = False
            

            for param in model.fc.parameters():
                param.requires_grad = True  

            model = model.to(self.device)       
        return model
    
    def second_phase_model(self):
        config = SimpleConfig('config.json')
        model_config = config['model']
        if model_config['type'] == 'resnet50':
            model = resnet50(weights = 'IMAGENET1K_V1')
            model.fc = nn.Sequential(
            nn.Linear(2048, 512),           
            nn.BatchNorm1d(512),            
            nn.ReLU(),
            nn.Dropout(0.5),                
            nn.Linear(512, self.num_classes))
            
            for param in model.parameters():
                param.requires_grad = False
            

            for param in model.fc.parameters():
                param.requires_grad = True  


            model = model.to(self.device)       
        return model

