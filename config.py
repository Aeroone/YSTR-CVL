######################################
############# Parameters #############
######################################
is_cuda = True # Turn to False if CPU used
img_root = "dataset/5pairs/train/images"   # "dataset/5pairs/train/images"
text_root = "dataset/5pairs/train/qa"
is_QApairs = False # If false, we use caption, if true, we use Q/A pairs
max_words_length = 9  # 13 for QA pairs, and 9 for captions
use_vgg = True  # Do not change this, we use vgg here for image encoding!


### test ###
img_test_root = "dataset/5pairs/test/images"   # "dataset/5pairs/train/images"
text_test_root = "dataset/5pairs/test/qa"
test_G_model = "model/G_epoch_1000.pth" # change this for the corresponding model!
test_num_workers = 1
test_result_root = "dataset/5pairs/test/test result"
############
######################################

#######################################################
### learning parameters for combined model training ###
#######################################################
Generator_model_filename = 'model/Generator_training_model.pth' # save the generator trained model
batch_size = 20 #64
num_workers = 4 # threads number for data loader
num_epochs = 1000
is_residual_module = True # use residual module in the generator
learning_rate = 0.0002
momentum = 0.5
lr_decay = 0.5
# --- attention model ---
is_attention = False
######################################

######################################
### visual-semantic text embedding ###
######################################
VSE_model_filename = 'model/VSE_training_model.pth'
pretrained_word_model = 'glove.6B.300d.txt'
VSE_batch_size = 64
VSE_num_workers = 4 # threads number for data loader
VSE_num_epochs = 300
VSE_embedding_dim = 300
VSE_learning_rate = 0.0002
VSE_margin  = 0.2
############################################################
############################################################