# YSTR-CVL

Instruction:

Please install python 2.7, pytorch, torchvision, torchtext (https://github.com/pytorch/text), 
PIL, NLTK before usage.


We use pretrained text-visual embedding for sentence encoder and "vgg" for image encoder.
"config": parameters to be used
If you have no GPU, set "is_cuda = False" in "config.py"


### Dataset preparation ###:

Open dataset file, open CLEVER file, and put corresonding files into "CLEVER/train" file.
(You can use 5pairs for trying, and change the route in the "config")

1. Train the text-visual embedding model for sentence embedding:
   (1) open "train_visual_embedding.py"
   (2) run it, and for the first time, it will automatically download the   
       "glove.6B.300d.txt" for word embedding
   (3) Finish this procedure, you will obtain the GRU/LSTM model for sentence embedding.
       The trained model will be saved in the "./model/VSE_training_model.pth"
2. Train the Combined model for the project
   (1) open "train.py"
   (2) run it, and for the first time, pytorch will automatically download the "vgg" models
   (3) Finish the procedure, you will obtain the trained generator model,
       The trained model will be saved in the "./model/Generator_training_model.pth"
       The result of fake images will be saved in the "result" file. 

### Current problems ###:

1. image reconstruction error addition
2. understand text augmentation methods
3. introduce attention models

