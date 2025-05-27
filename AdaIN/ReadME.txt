Original Code belongs to Anthony Ke, Hersh Vakharia, Issac Fung, Ali Baker
You can view their code here: https://github.com/hvak/adaIN-style-transfer/tree/main


Requirements:
torch
PIL 

The training set is already preloaded with 500 images for content and 500 images for style (images of landscapes & artwork)

Modifications I made to the original code:
Freezing parameters of VGG-19 network to ensure only decoder is optimized
Changing layers of VGG-19 from [1, 6, 1, 20] to [1, 6, 11, 20]
Style layers include: relu1_1 to relu4_1 instead of relu1_1 to relu5_1
Change mode of upsampling layers from nearest to bilinear
Addition of total loss and total loss plot, removal of average loss

How to use:
Hyperparameters in train.py are tuned to my optimal configuration 

Use the command below to run the code without any changes to the training set
python train.py

After the model is trained

Insert 1 or more content images into test_set/content and 1 or more style images into test_set/style

*If using more than 1 image, modify test.py to reflect the change

Use the command below to test the model:
python test.py


