######################################
############# Parameters #############
######################################
is_cuda = True # Turn to False if CPU used
img_root = "dataset/500-samples/train/images"   # "dataset/5pairs/train/images"
text_root = "dataset/500-samples/train/qa"
is_QApairs = False # If false, we use caption, if true, we use Q/A pairs
max_words_length = 9  # 13 for QA pairs, and 9 for captions
use_vgg = True  # Do not change this, we use vgg here for image encoding!

### test ###   /home/aeroone/Research/Model_v1/
img_test_root = "dataset/500-samples/test/images"   # "dataset/5pairs/train/images"
text_test_root = "dataset/500-samples/test/qa"
test_G_model_root = "dataset/500-samples/test/model" # change this for the corresponding model!
test_num_workers = 4
batchsize_test = 10
test_result_root = "dataset/500-samples/test/test result"
############
######################################

#######################################################
### learning parameters for combined model training ###
#######################################################
Generator_model_filename = 'model/Generator_training_model.pth' # save the generator trained model
batch_size = 30 #64
num_workers = 4 # threads number for data loader
num_epochs = 1000
is_residual_module = True # use residual module in the generator
learning_rate = 0.0001
momentum = 0.1
lr_decay = 0.2
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