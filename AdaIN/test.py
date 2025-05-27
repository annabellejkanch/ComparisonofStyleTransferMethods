import torch
import torchvision.utils as utils
from StyleTransfer import StyleTransfer
from PIL import Image
from ImageDataset import ImageDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from ImageDataset import DeNormalize
import time

if __name__ == '__main__':

    model = StyleTransfer()
    model.decoder.load_state_dict(torch.load("models/best_model.pth", map_location='cpu'))

    model.training = False
    model.eval()


    num_images = 1
    content = ImageDataset(flag='content', root_dir='./test_set/content', data_range=(0,num_images))
    style = ImageDataset(flag='style', root_dir='./test_set/style', data_range=(0,num_images))
    content_img = DataLoader(dataset=content, batch_size=1, shuffle=False)
    style_img = DataLoader(dataset=style, batch_size=1, shuffle=False)

    
    denormalizer = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    i = 0
    for content_batch, style_batch in zip(content_img, style_img):
        start_time = time.time()

        with torch.no_grad():
            decoded = model.forward(content_batch, style_batch)
        
        end_time = time.time()
        time_taken = end_time - start_time

        # Print the time taken for the current image
        print(f"Time taken to stylize image {i}: {time_taken:.2f} seconds")

        saved = decoded.clone().detach()
        saved = denormalizer(saved)

        utils.save_image(saved, "test_set/results/img" + str(i) + ".png")
        i+=1
