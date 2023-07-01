import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from utils_funcs import to_device, accuracy, conv_block

class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)
    
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

## Building the model
# We will extend `ImageClassificationBase` to develop the `ResNet9` 
# model which consist of `Residual Blocks` after every two CNN layer

class ResNet9(ImageClassificationBase):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            
            self.conv1 = conv_block(in_channels, 64)
            self.conv2 = conv_block(64, 128, pool=True)
            self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
            
            self.conv3 = conv_block(128, 256, pool=True)
            self.conv4 = conv_block(256, 512, pool=True)
            self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
            
            self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1), 
                                            nn.Flatten(), 
                                            nn.Dropout(0.2),
                                            nn.Linear(512, num_classes))
            
        def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.classifier(out)
            return out