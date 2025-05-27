#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import numpy as np
import imageio

#Device Class
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("No GPU available. Using CPU.")
    return device

#Neural Style Transfer Class
#Here we load & preprocess the image using the load_image class
#We load the VGG-19 model and freeze its parameters because we only want our generated image to be optimized
class NeuralStyleTransfer:
    def __init__(self, content_img, style_img, device):

        self.device = device
    
        self.content_img = self._load_image(content_img).to(device)
        self.style_img = self._load_image(style_img).to(device)
        
        self.vgg = models.vgg19(pretrained = True).features.to(device).eval()
        
        #Freezing VGG-19 parameters to ensure only generated image is optimized
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        #Content Layers: using a deeper layer for content to extract semantic information
        self.content_layers = ['conv4_2']
        #Style Layers: multiple lower layers used to extract style feature maps
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        
        #Normalization parameters, used in image preprocessing
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    #Class to load and preprocess the image
    def _load_image(self, image_path, size=512):
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    #Class to extracting the feature maps uses previously defined layers
    def extract_features(self, x):
        features = {}
        for name, layer in enumerate(self.vgg):
            x = layer(x)
            if f'conv{name//2 + 1}_{name % 2 + 1}' in self.content_layers + self.style_layers:
                features[f'conv{name//2 + 1}_{name % 2 + 1}'] = x
        return features

    #Class to compute content loss
    def compute_content_loss(self, gen_features, content_features):
        return nn.functional.mse_loss(gen_features, content_features)

    #Class to compute style loss using a gram matrix
    def compute_style_loss(self, gen_features, style_features):
        # Gram matrix computation
        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b * c, h * w)
            G = torch.mm(features, features.t())
            return G.div(b * c * h * w)
        
        style_loss = 0
        for name in self.style_layers:
            gen_gram = gram_matrix(gen_features[name])
            style_gram = gram_matrix(style_features[name])
            style_loss += nn.functional.mse_loss(gen_gram, style_gram)
        return style_loss
    
    #Class to compute total variation loss
    def total_variation_loss(self, img):
        return (torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + 
                torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])))

    #Class to extract content image and style image feature maps
    #Initializes the generated image using random noise
    def transfer_style(self, num_epochs=300, content_weight=1, style_weight=1e6, tv_weight=1e-6):
        
        content_features = self.extract_features(self.content_img)
        style_features = self.extract_features(self.style_img)
        
        generated_img = torch.randn(self.content_img.shape, requires_grad=True, device=self.device)
        
        #Used LBFGS optimizer as it worked better then Adam optimizer
        optimizer = optim.LBFGS([generated_img], max_iter=20)
        
        total_losses, content_losses, style_losses, tv_losses, gif_frames = [], [], [], [], []
        
        #Class to compute loss and optimize
        def closure():

            optimizer.zero_grad()
            
            #Clamping the generated image to ensure it's within specified range
            with torch.no_grad():
                generated_img.clamp_(-3, 3)
            
            #Extracting feature map of generated image
            gen_features = self.extract_features(generated_img)
            
            #Computing content loss between generated image and content image
            content_loss = self.compute_content_loss(gen_features[self.content_layers[0]], content_features[self.content_layers[0]])
            #Computing style loss between generated image and style image
            style_loss = self.compute_style_loss(gen_features, style_features)
            #Computing total variation loss between horizontal and vertical pixels in generated image
            tv_loss = self.total_variation_loss(generated_img)
            
            #Computing Total Loss
            total_loss = (content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss)
            
            #Backpropagation 
            total_loss.backward()
            
        
            total_losses.append(total_loss.item())
            content_losses.append(content_weight * content_loss.item())
            style_losses.append(style_weight * style_loss.item())
            tv_losses.append(tv_weight * tv_loss.item())
            
            return total_loss
        
        #Printing Epoch and its loss
        print(f"{'Epoch':^10}{'Total Loss':^15}{'Content Loss':^15}{'Style Loss':^15}{'TV Loss':^15}")
        print("-" * 70)
        
        #Optimization at each epoch
        start_time = time.time()
        for epoch in range(num_epochs):
            optimizer.step(closure)
            
            if epoch % 10 == 0:
                print(f"{epoch:^10}{total_losses[-1]:^15.4f}{content_losses[-1]:^15.4f}"f"{style_losses[-1]:^15.4f}{tv_losses[-1]:^15.4f}")

                with torch.no_grad():
                    generated_img_np = generated_img.squeeze(0).cpu().detach()
                    generated_img_np = generated_img_np * self.std.view(3, 1, 1) + self.mean.view(3, 1, 1)
                    generated_img_pil = transforms.ToPILImage()(torch.clamp(generated_img_np, 0, 1))
                    gif_frames.append(generated_img_pil)

        #Computing Time
        total_time = time.time() - start_time
        print(f"\nTotal optimization time: {total_time:.2f} seconds")
        print(f"Average time per epoch: {total_time/num_epochs:.4f} seconds")

        gif_frames[0].save('style_transfer.gif', save_all=True, append_images=gif_frames[1:], duration = 100, loop=0)
        print("Gif saved'")

        # Plotting
        plt.figure(figsize=(15, 5))
        
        #Total Loss v Epoch Plot
        plt.subplot(131)
        plt.plot(total_losses)
        plt.title('Total Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        #Content Loss vs Epoch Plot
        plt.subplot(132)
        plt.plot(content_losses)
        plt.title('Content Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        #Total Variation Loss vs Epoch Plot
        plt.subplot(133)
        plt.plot(tv_losses)
        plt.title('Total Variation Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('loss_plots.png')
        
        #Saving the generated image
        generated_img_np = generated_img.squeeze(0).cpu().detach()
        generated_img_np = generated_img_np * self.std.view(3, 1, 1) + self.mean.view(3, 1, 1)
        generated_img_pil = transforms.ToPILImage()(torch.clamp(generated_img_np, 0, 1))
        generated_img_pil.save('generated_image.png')
        
        return generated_img_pil

def main():
    
    device = get_device()

    try:
        style_transfer = NeuralStyleTransfer('content.jpg', 'style.jpg', device)
        result = style_transfer.transfer_style()
        print("Neural Style Transfer completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
    