import torch
import torch.optim as optim
from ImageDataset import ImageDataset as ImgDataset
from StyleTransfer import StyleTransfer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms, utils
from tqdm import tqdm
import statistics
import torch.nn as nn


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        print("Could not find GPU, using CPU instead")

    # Hyperparameters
    lr = 1e-6
    num_epochs = 100
    style_weight = 30
    content_weight = 1
    batch_size = 8

    data_size = 500  
    train_content_data = ImgDataset(flag='train', root_dir='./train_set/content', data_range=(0, data_size))
    train_style_data = ImgDataset(flag='train', root_dir='./train_set/style', data_range=(0, data_size))
    train_content = DataLoader(dataset=train_content_data, batch_size=batch_size, shuffle=True)
    train_styles = DataLoader(dataset=train_style_data, batch_size=batch_size, shuffle=True)

    
    model = StyleTransfer().to(device)
    model.training = True

    optimizer = optim.Adam(model.parameters(), lr=lr)

    tqdm_tot = min(len(train_content), len(train_content))
    losses = []
    content_losses = []
    style_losses = []

    for i in range(num_epochs):
        print('-----------------Epoch = %d-----------------' % (i+1))
        running_loss = []
        running_content_loss = []
        running_style_loss = []
        
        for content_batch, style_batch in tqdm(zip(train_content, train_styles), total=tqdm_tot):
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            decoded, content_loss, style_loss = model.forward(content_batch, style_batch)
            
            total_loss = (content_weight * content_loss + style_weight * style_loss)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss.append(total_loss.item())
            running_content_loss.append(content_loss.item())
            running_style_loss.append(style_loss.item())
        
        epoch_loss = statistics.mean(running_loss)
        epoch_content_loss = statistics.mean(running_content_loss)
        epoch_style_loss = statistics.mean(running_style_loss)
        
        print("Total Loss = ", epoch_loss)
        print("Content Loss = ", epoch_content_loss)
        print("Style Loss = ", epoch_style_loss)

        losses.append(epoch_loss)
        content_losses.append(epoch_content_loss)
        style_losses.append(epoch_style_loss)

        torch.save(model.decoder.state_dict(), "models/model_epoch" + str(i) + ".pth")
    
    # Plot total loss, content loss, and style loss
    x = np.arange(len(losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, losses, label="Total Loss")
    plt.plot(x, content_losses, label="Content Loss")
    plt.plot(x, style_losses, label="Style Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Losses over Epochs")
    plt.savefig("loss_plot.png")
