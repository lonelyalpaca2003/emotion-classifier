import torch
from torch.amp import autocast
import matplotlib.pyplot as plt
from config_parser import SimpleConfig

config = SimpleConfig('config.json')

class Trainer:
   def __init__(self, model, criterion, optimizer, scheduler  = None, device = 'cuda', early_stopping = None):
      self.model = model
      self.criterion = criterion
      self.optimizer = optimizer
      self.device = device
      self.scheduler = scheduler
      self.early_stopping = early_stopping
      self.scaler = torch.amp.GradScaler()

      self.train_losses = []
      self.train_accs = []
      self.val_losses = []
      self.val_accs = []


   def train_model(self, dataloader):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in dataloader:
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with autocast(device_type = self.device):
              prediction = self.model(inputs)
              batch_loss = self.criterion(prediction, labels)

            total_loss_train += batch_loss.item()

            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc = (torch.argmax(prediction, dim=1) == labels).sum().item()
            total_acc_train += acc

        avg_loss_train = total_loss_train / len(dataloader)
        avg_acc_train = (total_acc_train / len(dataloader.dataset)) * 100




        return avg_loss_train, avg_acc_train

   def validate(self, dataloader):
      total_loss_val = 0
      total_acc_val = 0
      with torch.no_grad():
          for inputs, labels in dataloader:
              inputs, labels = inputs.to(self.device), labels.to(self.device)

              with autocast(device_type = self.device):

                prediction = self.model(inputs)
                val_loss = self.criterion(prediction, labels)

              total_loss_val += val_loss.item()

              val_acc = (torch.argmax(prediction, dim=1) == labels).sum().item()
              total_acc_val += val_acc

      avg_loss_val = total_loss_val / len(dataloader)
      avg_acc_val = (total_acc_val / len(dataloader.dataset)) * 100

      return avg_loss_val, avg_acc_val

   def fit(self, train_dataloader, val_dataloader, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_model(train_dataloader)
            val_loss, val_acc = self.validate(val_dataloader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']

            print(f'''Epoch: {epoch + 1}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}, Current LR: {current_lr}''')
            if self.early_stopping.early_stop(val_loss):
              print(f"Early stopping initiated at epoch {epoch + 1}")
              break

   def plot_artifacts(self):
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
        axs[0].plot(self.train_losses, label = "training_loss")
        axs[0].plot(self.val_losses, label = "validation loss")
        axs[0].set_title("Training and validation loss over epochs")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].set_ylim([0,2])
        axs[0].legend()


        axs[1].plot(self.train_accs, label = "training accuracy")
        axs[1].plot(self.val_accs, label = "validation accuracy")
        axs[1].set_title("Training and validation accuracy over epochs")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_ylim([0,100])
        axs[1].legend()

        plt.show()
