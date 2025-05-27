# Comparison of Style Transfer Methods
Malevich Neural Style Transfer with VGG-19 | Van Gogh Neural Style Transfer with VGG-19
:-------------------------:|:-------------------------:
 ![Malevich Style Transfer Gif](https://github.com/user-attachments/assets/1fe75d9b-49c2-42c1-9cf4-1b82eb6c3add) |   ![Van Gogh Style Transfer Gif](https://github.com/user-attachments/assets/4faa3db1-ebdb-4897-9333-1c2c20653296)


## Abstract
Style Transfer, an interesting domain in computer  vision, combines the content of one image with the style of another to create artistic outputs. Today there are many style transfer techniques, each with their own strengths & limitations. Which is why it is essential to compare methods and identify where these techniques can improve and the specific tasks that they would be useful for. This project presents a comparative analysis of traditional & modern methods of style transfer techniques. The style transfer techniques focused on in this project are Neural Style Transfer (NST) technique by Gatys et al [11] and the Adaptive Instance Normalization (AdaIN) technique by Huang et al [2]; both utilizing the VGG-19 neural network for feature extraction. Quantitative metrics such as total loss, visual quality, alongside human evaluation, highlight the strengths and limitations of each method. The findings revealed that NST excels in intricate style representation, with a low total loss; but requires significant time for iterative optimization during training. Whereas AdaIN offers greater adaptability and faster stylization; but fails at accurately translating style while maintaining content, accounting for its larger total loss values.  

## Neural Style Transfer with VGG-19 Architecture
The architecture of my NST model is centered around the VGG-19 network. The content layer is conv4_2, and the style layers include conv1_1, conv2_1, conv3_1, conv4_1, conv5_1. The style layers are the same ones utilized in Gatys et al’s implementation of NST. Only the convolutional and max-pooling layers of the VGG-19 network are used, and all parameters of the network are frozen to ensure the network works exclusively for feature extraction. 
The process begins by initializing a generated image with random noise. During each epoch, the VGG-19 network will extract feature maps for the content, style, and generated image. These feature maps will then be used to compute the following loss functions: content loss, style loss, total variation loss, and total loss. Key hyperparameters of the model include content weight, style weight, total variation weight, and learning rate. These first three hyperparameters mentioned control the contribution of each loss term to the total loss. The 
LBFGS optimizer is then employed to minimize the total loss, refining the generated image with each epoch

## AdaIN with VGG-19 Architecture
Like my NST model, the architecture for the AdaIN model is also centered around the VGG-19 network. Only the convolutional and max-pooling layers are used from the 
VGG-19 network, and its parameters are frozen so that only the decoder is optimized. In the VGG-19 encoder, the first 21 layers of the network are used. The content layer includes relu4_2 and the style layers include relu1_1, relu2_1, relu3_1, and relu4_1, this differs from the style layers used in GitHub repository. Forward hooks are utilized to capture the feature maps at the layers mentioned above. Once the feature maps are extracted, they are passed through the AdaIN class which will then normalize the content feature maps mean and standard deviation to match that of the style feature map. Once the image is transformed using the AdaIN class, it is then passed to the decoder, which is the inverse of the 
VGG-19 encoder. The convolutional layers of the decoder use a padding of 1 and reflection padding which helps preserve edges. The ReLU layer introduces non-linearity into the model. The Upsampling layer increased the spatial dimensions; here I decided to change mode from nearest to bilinear, because having the mode as nearest resulted in blocky generated image. After the decoder decodes the stylized image, it is returned and then passed back to the 
encoder to compute content loss (2), style loss (3), and total loss (4) (The GitHub repository had used average loss to evaluate the model, but for accurate comparison between 
both models I used total loss).  Key hyperparameters of the model include content weight, style weight, and learning rate. These hyperparameters control the contribution of each loss term to the total loss. The Adam optimizer then minimizes the calculated total loss and optimizes the parameters of only the decoder.  





