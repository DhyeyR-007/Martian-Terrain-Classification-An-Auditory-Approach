# %%

### This .py file is converted from original .ipynb 
### GOOGLE DRIVE LINK TO THE WHOLE NOTEBOOK (WITH RESULTS OF TRAJECTORY 12  IN THE OUTPUTS OF NOTEBOOK CELLS): https://drive.google.com/file/d/1yXovFxhFuHE9lsN2RNW9yt8MH6dC4gFB/view


import os
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import *
from keras.models import Sequential
from keras.optimizers import Adam, Adadelta
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.datasets import mnist


from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform,he_uniform

# from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer, InputSpec,Dropout
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model,normalize

from sklearn.metrics import roc_curve,roc_auc_score

# %%
!pip install librosa
!pip install pydub

# %%
"""
Trajectory No.12(run12 according to dataset provided by Jannik Zürn, Wolfram Burgard, Abhinav Valada
                                                          Self-Supervised Visual Terrain Classification From Unsupervised Acoustic Feature Learning
                                                          IEEE Transactions On Robotics (T-RO), Vol. 37, No. 2, Pp. 466-481, 2019.
                                                          Bibtex)
"""

# FIVE TERRAINS:
    # 1. GRAVEL
    # 2. ASPHALT
    # 3. GRASS
    # 4. COBBLE
    # 5. PL {Parking Lot}


import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt



# converts sec to hr:min:sec
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)



# Load the mixed audio clip {Change filename here}
filename1 = '/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/GRAVEL_R12/GRAVEL_joined_R12.wav'

filename2 = '/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/ASPHALT_R12/ASPHALT_joined_R12.wav'

filename3 = '/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/GRASS_R12/GRASS_joined_R12.wav'

filename4 = '/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/COBBLE_R12/R12_cobble 0.wav'

filename5 = '/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/PL_R12/R12_PL 0.wav'





File_ = [filename1, filename2, filename3, filename4, filename5]

TimeList ={}

for i in File_:
    y, sr = librosa.load(i,sr=44100)
    plt.figure(figsize=(50, 10))
    librosa.display.waveshow(y, sr=sr)

    if 'GRAVEL' in i:
        plt.title('Signal_Gravel', fontsize=60)
        gravel = librosa.get_duration(y=y, sr=sr)
        TimeList[gravel] = [y,'GRAVEL']
        print("Duration gravel: {0} min == {1}secs.".format(convert(gravel),gravel))

    if 'COBBLE' in i:
        plt.title('Signal_Cobbelstone',fontsize=60)
        cobbelstone = librosa.get_duration(y=y, sr=sr)
        TimeList[cobbelstone] = [y,'COBBLE']
        print("Duration cobbelstone: {0} min == {1}secs. ".format(convert(cobbelstone),cobbelstone))


    if 'ASPHALT' in i:
        plt.title('Signal_Asphalt',fontsize=60)
        asphalt = librosa.get_duration(y=y, sr=sr)
        TimeList[asphalt] = [y,'ASPHALT']
        print("Duration asphalt: {0} min == {1}secs. ".format(convert(asphalt),asphalt))


    if 'GRASS' in i:
        plt.title('Signal_Grass',fontsize=60)
        grass = librosa.get_duration(y=y, sr=sr)
        TimeList[grass] = [y,'GRASS']
        print("Duration grass: {0} min == {1}secs. ".format(convert(grass),grass))


    if 'PL' in i:
        plt.title('Signal_PL',fontsize=60)
        PL = librosa.get_duration(y=y, sr=sr)
        TimeList[PL] = [y,'PL']
        print("Duration PL: {0} min == {1}secs. ".format(convert(PL),PL))
        









# %%
"""
Trajectory No.12(run12 according to dataset provided by Jannik Zürn, Wolfram Burgard, Abhinav Valada
                                                          Self-Supervised Visual Terrain Classification From Unsupervised Acoustic Feature Learning
                                                          IEEE Transactions On Robotics (T-RO), Vol. 37, No. 2, Pp. 466-481, 2019.
                                                          Bibtex)
"""

# FIVE TERRAINS:
    # 1. GRAVEL
    # 2. ASPHALT
    # 3. GRASS
    # 4. COBBLE
    # 5. PL {Parking Lot}


import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt



# converts sec to hr:min:sec
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)



# Load the mixed audio clip {Change filename here}
filename1 = '/content/drive/MyDrive/Untitled folder/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/Audio files/Rover_Audio.wav'
File_ = [filename1]


for i in File_:
      y, sr = librosa.load(i,sr=None)
      plt.figure(figsize=(50, 10))

      librosa.display.waveshow(y, sr=sr)
      plt.title('Original Mixed Signal',fontsize=60)



      # ps = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)

      # plt.figure(figsize=(10, 4))
      # librosa.display.specshow(ps, x_axis='time')
      # plt.colorbar()
      # plt.title('MFCC_Gravel')
      # plt.tight_layout()
      # plt.show()



    



    # if 'COBBLE' in i:
    #     plt.title('Signal_Cobbelstone',fontsize=60)
    #     cobbelstone = librosa.get_duration(y=y, sr=sr)
    #     TimeList[cobbelstone] = [y,'COBBLE']
    #     print("Duration cobbelstone: {0} min == {1}secs. ".format(convert(cobbelstone),cobbelstone))


    # if 'ASPHALT' in i:
    #     plt.title('Signal_Asphalt',fontsize=60)
    #     asphalt = librosa.get_duration(y=y, sr=sr)
    #     TimeList[asphalt] = [y,'ASPHALT']
    #     print("Duration asphalt: {0} min == {1}secs. ".format(convert(asphalt),asphalt))


    # if 'GRASS' in i:
    #     plt.title('Signal_Grass',fontsize=60)
    #     grass = librosa.get_duration(y=y, sr=sr)
    #     TimeList[grass] = [y,'GRASS']
    #     print("Duration grass: {0} min == {1}secs. ".format(convert(grass),grass))


    # if 'PL' in i:
    #     plt.title('Signal_PL',fontsize=60)
    #     PL = librosa.get_duration(y=y, sr=sr)
    #     TimeList[PL] = [y,'PL']
    #     print("Duration PL: {0} min == {1}secs. ".format(convert(PL),PL))
        










# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
"""
# AUDIO DATA AUGMENTATION

"""

# %%
import librosa
import numpy as np
sr = 44100


# Define data augmentation techniques
def time_stretch(audio):
    rate = np.random.uniform(low=0.8, high=1.2)  # 0.8: This makes the sound deeper but we can still hear 'off' ; 1.2: High Frequency 
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio):
    n_steps = np.random.randint(low=-3, high=3)
    return librosa.effects.pitch_shift(audio, sr=44100, n_steps=n_steps)

def add_noise(audio):
    noise = np.random.normal(0, 0.1, len(audio))
    alpha = np.random.uniform(low=0.01, high=0.1)
    return audio + alpha * noise

def Equal_Time(audio,A):
    # Determine current duration in seconds
    current_duration = librosa.get_duration(y=audio, sr=sr)

    # Determine target duration in seconds
    target_duration = A  # 7 minutes in seconds

    # Determine number of samples to add
    samples_to_add = int((target_duration - current_duration) * sr)

    # Apply data augmentation to extend audio
    audio_extended = audio.copy()
    aug = [time_stretch, pitch_shift, add_noise]
    while len(audio_extended) < (target_duration * sr):
        # Apply time stretching
        
        augmentation_fn = np.random.choice(aug)
        audio_extended = np.concatenate([audio_extended, augmentation_fn(audio)])
        # Ensure audio doesn't exceed target duration
        audio_extended = audio_extended[:int(target_duration * sr)]
    
    return audio_extended




# %%
"""
# AUDIO CLIP EQUALIZATION
"""

# %%
## COLLECTING ALL THE EQUAL DURATION SEPARATE TERRAIN AUDIO CLIPS IN A DICTIONARY

DictiZ = {}


for key in TimeList.keys():
    A = max(TimeList)
    y = TimeList[key][0]
    name = TimeList[key][1]

    sr = 44100



    
    if name == 'GRAVEL':
      DictiZ['gravel'] = Equal_Time(audio=y,A=A)
    elif name == 'ASPHALT':
      DictiZ['asphalt'] = Equal_Time(audio=y,A=A)
    elif name == 'COBBLE':
      DictiZ['cobbelstone'] = Equal_Time(audio=y,A=A)
    elif name == 'GRASS':
      DictiZ['grass'] = Equal_Time(audio=y,A=A)
    elif name == 'PL':
      DictiZ['PL'] = Equal_Time(audio=y,A=A)




# %%
DictiZ

# %%
import numpy as np
import soundfile as sf

sr = 44100 # Hz

filename_= "/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/R12_Testing"   

os.mkdir(filename_)

for key in DictiZ.keys():
    print(key)
    ipd.display(ipd.Audio(DictiZ[key], rate=sr))
    filename = filename_ + "/" + key + "_Final" + '.wav'
    sf.write(filename, DictiZ[key], sr)


# %%
# Function to remove alrady created filled or unfilled directories
import shutil
for key in DictiZ.keys():
    path = "/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R04/R04_Testing/" + key+'_chunks'     
    shutil.rmtree(path)

# %%
####################################### ZUHRN AUDIO ###########################################

## Splitting the above segregated audios into chunks for training and testing data preparation ##

from pydub import AudioSegment
from pydub.utils import make_chunks



for key in DictiZ.keys():
    filename = "/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/R12_Testing/" + key +"_Final" + '.wav'
    myaudio = AudioSegment.from_wav( filename, "wav") 
    chunk_length_ms = 500 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) # Make chunks

    lisk = []

    filename__ = "/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/R12_Testing/" + key+'_chunks'     
    os.mkdir(filename__)

    for i,chunk in enumerate(chunks):
       chunk_name = "New_bit{0}.wav".format(i)
       print ("exporting", chunk_name)

       filename__ = "/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/R12_Testing/" + key+'_chunks' +'/chunk{0}.wav'.format(i)   
       chunk.export(filename__, format="wav")
       lisk.append(filename__)

    
    DictiZ[key] = lisk
    print("---------------------------")

                                                               





# %%
## Sanity Check before train/test dataset creation

for key in DictiZ.keys():
    print(key)
    print(len(DictiZ[key]))
    print('-------------------------')


# %%
## Sanity Check
DictiZ['gravel'][0]

# %%
!pip install tqdm

# %%
## CONVERTING THE OBTAINED AUDIO CLIPS INTO MFCC FORMAT (DIMENSIONALLY REDUCING THEM ESSENTIALLY) FOR BETTER FEATURE EXTRACTION


import numpy as np
from tqdm import tqdm
import keras
import librosa
import librosa.display
import random

import warnings
warnings.filterwarnings('ignore')


# # I am using dictionary here so that i can assign labels to each audio file easily instead of manual labelling
Dicti_Fin = {}

for key in DictiZ.keys():
    temp=[]

    for j in range(len(DictiZ[key])):

      

      y, sr = librosa.load(DictiZ[key][j])
    
      # ps = librosa.feature.melspectrogram(y=y, sr=sr) 
      ps = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)

    

      if ps.shape[1] != 22 or ps.shape[0] != 128 :
        continue
        



      ps = ps.reshape(128, -1, 1)

      temp.append(ps)
    
    Dicti_Fin[key] = temp
    print("Done {0}".format(key))





# %%
## Sanity check
len_temp = []
for key in Dicti_Fin.keys():
    print(len(Dicti_Fin[key]))
    len_temp.append(len(Dicti_Fin[key]))
    print(Dicti_Fin[key][0].shape)
    break

# %%
from itertools import repeat

limi = int(len_temp[0] * 0.8)


lisk_train = []
lisk_train_L = []
for i in Dicti_Fin.keys():
      lisk_train.append(Dicti_Fin[i][0:limi]) # list with training data audio files (each one is a chunk)
      lisk_train_L.append([i] * len(lisk_train[0]))    # list with training data labels


lisk_test = []
lisk_test_L = []
for i in Dicti_Fin:
      lisk_test.append(Dicti_Fin[i][limi:])  # list with testing data audio files (each one is a chunk)
      lisk_test_L.append([i] * len(lisk_test[0]))  # list with testing data labels


# %%
"""
# Model Architecture, Triplet Loss & Layers
"""

# %%
# here we define the layers of neural net
# here i have only taken few layers, since I am currently decding upon the layers as a hyperparameter, to tune
# when i use my set approaches( approach1 and 2) on the whole dataset
# this is just to handle first four minutes of data, cuz that much only the google colab was able to process without timeout

"""
NOTE:   I AM WORKING ON IMPROVING THIS LAYER PART IN MY APPROACH 2 DEVELOPMENT THIS MODEL IS JUST FOR YOUR SIMPLICITY JUST TO UNDERSTAND HOW THE SYSTEM IS WORKING

"""

def network_architecture(input_shape, embeddingsize):
    '''
    Input : 
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our picture   
    '''
     # Convolutional Neural Network
    network = Sequential()

    ### i tried using automatic hyperparameter tuning, please dont erase this 

#     network.add(Conv2D(128, (3,3), activation='relu',
#                      input_shape=input_shape,
#                      kernel_initializer='he_uniform',
#                      kernel_regularizer=l2(2e-4)))
    # network.add(MaxPooling2D())
#     A = hp.In
#     hp_units1 = hp.Int('units1',min_value=32, max_value = 512, step= 32)



    network.add(Conv2D(128, (3,3), activation='relu',
                     input_shape=input_shape,
                     kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D(2,strides=2))
    
    # network.add(Conv2D(64, (3,3), activation='relu',
    #                  input_shape=input_shape,
    #                  kernel_initializer='he_uniform',
    #                  kernel_regularizer=l2(2e-4)))
    # network.add(MaxPooling2D(2,strides=2))


    network.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    network.add(MaxPooling2D(2,strides=2))

    network.add(Flatten())
    
    network.add(Dense(128, activation='relu',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'))
    
    network.add(Dense(embeddingsize, activation=None,
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'))
    
    #Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    return network

# %%
# calculation of triplet loss
# this function can be modified to calculte contrastive loss and quadruplet loss quickly without many errors

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

# %%
# this function is the starting point for any training that happend for the neeural network
# the input audio is first passed through Input function for making it ready to e fed into the network
# then the inputs are passed into the neural net model which generates embeddings, nw these embeddings acan be use to calculate the distance 
# between the positive, negative and/or anchor, which can be used to compute triplet loss
# the last layer of my model architecture basically calculates triplet loss( that's what you see as output 'loss' while training)

def Model_Architecture_Start(input_shape, network, margin=0.2):
    '''
        Input: 
          network --> Neural network to train outputing embeddings
          input_shape --> shape of input audio
          margin --> minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    
    '''
     # Define the tensors for the three input audios
    anchor_ip = Input(input_shape, name="anchor_input")
    positive_ip = Input(input_shape, name="positive_input")
    negative_ip = Input(input_shape, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three audio files
    encoded_a = network(anchor_ip)
    encoded_p = network(positive_ip)
    encoded_n = network(negative_ip)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_ip,positive_ip,negative_ip],outputs=loss_layer)
    
    # return the model
    return network_train

# %%
## Driver code for calling out the neural net architecture
h_,w_,c_ = 128,22,1

row, col,channel = h_, w_, c_
input_shape = (row, col, channel)



network = network_architecture(input_shape,embeddingsize=5)
network_train = Model_Architecture_Start(input_shape,network)
optimizer = Adadelta(lr = 0.01)
network_train.compile(optimizer=optimizer)
network_train.summary()
print(network_train.metrics_names)
n_iteration=0




# %%
! pip install keras_visualizer

# %%
pip install keras-visualizer --upgrade

# %%
pip install visualkeras

# %%
from keras_visualizer import visualizer


import visualkeras
visualkeras.layered_view(network,legend=True)

# %%
## SANITY CHECK

featured_img = network.predict(np.ones((1,128,22,1)))
print(featured_img)

# %%
nb_classes =  len(TimeList)
from tensorflow.keras.utils import to_categorical

def random_triplets(batch_size,s='train'):
    """
    Creating triplets with random strategy
    Input:
    batch_size --> integer 

    Output:
    triplets --> list containing 3 tensors A,P,N of shape (batch_size,w,h,c)

    """
    if s == 'train':
        X = lisk_train
        y = lisk_train_L
    elif s == 'test':
        X = lisk_test
        y = lisk_test_L

        

    m = len(X[0])  # bactches = N
    w, h, c = X[0][0].shape  # W,H,C
    
    # initialize result
    triplets=[np.zeros((batch_size,w, h,c)) for i in range(3)]
    # triplets=[np.zeros((h, w,c)) for i in range(3)]



    for i in range(batch_size):
        #Pick one random class for anchor
        anchor_class = np.random.randint(0, nb_classes)
        nb_sample_available_for_class_AP = len(X[anchor_class])
        


        #Pick two different random pics for this class => A and P
        [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)
      
        #Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1,nb_classes)) % nb_classes
        nb_sample_available_for_class_N = len(X[negative_class])
        


        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i] = X[anchor_class][idx_A]
        triplets[1][i] = X[anchor_class][idx_P]
        triplets[2][i] = X[negative_class][idx_N]    # X[negative_class][idx_N,:,:,:]


    return triplets



# %%
# I have configured my network with regular triplets instead of hard triplets(which account for false pos. or false neg.)
## SANITY CHECK ##
triplets = random_triplets(2,s='test')

print("Checking batch width, should be 3 : ",len(triplets))
print("Shapes in the batch A:{0} P:{1} N:{2}".format(triplets[0].shape, triplets[1].shape, triplets[2].shape))



# %%

# distance computation

def compute_dist(a,b):
    return np.sum(np.square(a-b))



# function for gathering triplets which might prevent false negatives and false positives 
# hard triplets are the triplets where the negative is closer to the anchor than the positive, d(A,N)<d(A,P)

def hard_triplets(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="train"):
    """
    
    Input
    draw_batch_size --> integer : number of initial randomly taken samples   
    hard_batchs_size --> interger : select the number of hardest samples to keep
    norm_batchs_size --> interger : number of random samples to add

    Output
    triplets --> list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    if s == 'train':
        X = lisk_train
    else:
        X = lisk_test

    m = len(X[0])  # bactches = N
    w, h, c = X[0][0].shape  # W,H,C
    
    
    #Step 1 : pick a random batch to study
    studybatch= random_triplets(draw_batch_size,s)
    
    #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))
    
    #Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])
    
    #Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)
    
    #Sort by distance (high distance first) and take the 
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
    
    #Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)
    
    selection = np.append(selection,selection2)
    
    triplets = [studybatch[0][selection,:,:,:], studybatch[1][selection,:,:,:], studybatch[2][selection,:,:,:]]
    
    return triplets

# %%
## SANITY CHECK ##


hardtriplets = hard_triplets(50,1,1,network,s='train')
print("Shapes in the hardbatch A:{0} P:{1} N:{2}".format(hardtriplets[0].shape, hardtriplets[1].shape, hardtriplets[2].shape))


# %%
import math
def compute_probs(network,X,Y):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class
    
    Returns
        probs : array of shape (m,m) containing distances
    
    '''
    m = X.shape[0]
    nbevaluation = int(m*(m-1)/2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))
    
    #Compute all embeddings for all pics with current network
    embeddings = network.predict(X)
    
    size_embedding = embeddings.shape[1]
    
    #For each pics of our dataset
    k = 0
    for i in range(m):
            #Against all other images
            for j in range(i+1,m):
                #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
                probs[k] = -compute_dist(embeddings[i,:],embeddings[j,:])
                if (Y[i]==Y[j]):
                    y[k] = 1
                    #print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
                else:
                    y[k] = 0
                    #print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
                k += 1
    return probs,y
#probs,yprobs = compute_probs(network,x_test_origin[:10,:,:,:],y_test_origin[:10])



def compute_metrics(probs,yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)
    
    return fpr, tpr, thresholds,auc



def compute_interdist(network):
    '''
    Computes sum of distances between all classes embeddings on our reference test image: 
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings
        
    Returns:
        array of shape (nb_classes,nb_classes) 
    '''
    res = np.zeros((nb_classes,nb_classes))
    
    ref_images = np.zeros((nb_classes,img_rows,img_cols,1))
    
    #generates embeddings for reference images
    for i in range(nb_classes):
        ref_images[i,:,:,:] = Dataset_Test[i][0,:,:,:]
    ref_embeddings = network.predict(ref_images)
    
    for i in range(nb_classes):
        for j in range(nb_classes):
            res[i,j] = compute_dist(ref_embeddings[i],ref_embeddings[j])
    return res

def draw_interdist(network,n_iteration):
    interdist = compute_interdist(network)
    
    data = []
    for i in range(nb_classes):
        data.append(np.delete(interdist[i,:],[i]))

    fig, ax = plt.subplots()
    ax.set_title('Embeddings distance from another: No. of iterations {0}'.format(n_iteration))
    ax.set_ylim([0,10])
    plt.xlabel('Classes (labelled as numbers)')
    plt.ylabel('Distance')
    ax.boxplot(data,showfliers=False,showbox=True)
    locs, labels = plt.xticks()
    plt.xticks(locs,np.arange(nb_classes))

    plt.show()
    
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1],idx-1
    else:
        return array[idx],idx
    
def draw_roc(fpr, tpr,thresholds):
    #find threshold
    targetfpr=0.01
    _, idx = find_nearest(fpr,targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]
    
    
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr,'green')
    plt.title('Area under the curve : {0:.3f}\nSensitivity : {2:.1%}\nFalse Positive Rate={1:.0e}\nThreshold={3})'.format(auc,targetfpr,recall,abs(threshold) ))
    # show the plot
    plt.show()

# %%
Dataset_Test = []
for i in range(len(lisk_test)):
    temp = None
    temp = array(lisk_test[i])
    Dataset_Test.append(temp)


# %%
import tensorflow as tf

new_lisk_test = []
for i in range(len(lisk_test)):
  for j in range(len(lisk_test[i])):
    new_lisk_test.append(lisk_test[i][j])


y = array(new_lisk_test)
y_tensor = tf.convert_to_tensor(y, dtype=new_lisk_test[0].dtype) 

# new_y_tensor = tf.transpose(y_tensor, perm=[0, 3, 1, 2])
new_y_tensor = y_tensor # (88,128,128,1)  ### test dataset converted into tensor


# %%
new_lisk_test_L = []
for i in range(len(lisk_test_L)):
  for j in range(len(lisk_test_L[i])):
    new_lisk_test_L.append(lisk_test_L[i][j])

y_L = array(new_lisk_test_L)
new_y_tensor_L = tf.convert_to_tensor(y_L)  ### test dataset labels converted into tensor

# %%
# Testing on an untrained network
img_rows, img_cols = 128, 22


probs,yprob = compute_probs(network,new_y_tensor,new_y_tensor_L)
fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
draw_roc(fpr, tpr,thresholds)
draw_interdist(network,n_iteration)

# %%
Dataset_Train = []
for i in range(len(lisk_train)):
    temp = None
    temp = np.array(lisk_train[i])
    Dataset_Train.append(temp)


# %%
import tensorflow as tf

new_lisk_train = []
new_lisk_train_L = []

for i in range(len(lisk_train)):
  for j in range(len(lisk_train[i])):
    new_lisk_train.append(lisk_train[i][j])
    new_lisk_train_L.append(lisk_train_L[i][j])


y = array(new_lisk_train)
train_tensor = tf.convert_to_tensor(y, dtype=y.dtype)  ### train dataset converted into tensor

y_L = array(new_lisk_train_L)
train_tensor_L = tf.convert_to_tensor(y_L)  ### train dataset labels converted into tensor



# %%

def AudioDist(network, images, refidx=0):
    '''
    Evaluate some audio data vs some samples in the test set
        image must be of shape(1,w,h,c)
    
    Returns
        scores : 
    
    '''
 
    _, w,h,c = Dataset_Test[0].shape
    nbimages=images.shape[0]
    
    #generates embedings for given images
    image_embedings = network.predict(images)


    # generates embedings for reference images
    ref_images = np.zeros((nb_classes,w,h,c))
    for i in range(nb_classes):
        ref_images[i,:,:,:] = Dataset_Test[i][refidx,:,:,:]
    ref_embedings = network.predict(ref_images)
            


    for i in range(nbimages):       
        for ref in range(nb_classes):
            # Compute distance between this images and references
            dist = compute_dist(image_embedings[i,:],ref_embedings[ref,:])
            print("Distance from class {0} = {1}".format(ref,dist))

    print("----------------------------------")
            



# %%
nb_classes =  len(TimeList)

for i in range(nb_classes):
  print("Class {0}".format(i))
  AudioDist(network,np.expand_dims(Dataset_Test[i][0,:,:,:],axis=0))

# %%
"""
# Random Triplets
"""

# %%
"""
## Training


"""

# %%

print("Starting training process!")
print("-------------------------------------")


triplets_train = random_triplets(6000,s='train')

# hardtriplets = hard_triplets(5000,1000,4000,network,s='train')


loss = network_train.fit(triplets_train ,batch_size= 32, epochs=100, validation_split=0.2)

        

# %%
from matplotlib import pyplot as plt
# import ma
plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('Losses (Random triplets)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'], loc='upper right')
plt.show()

# %%
# save best random triplet weights
network_train.save_weights('/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/Weights_R12_RANDOM/')

# %%
"""
## Evaluation Metrics
"""

# %%
# Testing on an trained network
img_rows, img_cols = 128, 22


probs,yprob = compute_probs(network,new_y_tensor,new_y_tensor_L)
fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
draw_roc(fpr, tpr,thresholds)
draw_interdist(network,n_iteration)

# %%
nb_classes = len(TimeList)
for i in range(nb_classes):
  print("Class {0}".format(i))
  AudioDist(network,np.expand_dims(Dataset_Test[i][0,:,:,:],axis=0))

# %%
"""
# Hard Triplets
"""

# %%
"""
## Training
"""

# %%

print("Starting training process!")
print("-------------------------------------")




hardtriplets = hard_triplets(7000,2000,5000,network,s='train')


loss_ht = network_train.fit(hardtriplets,batch_size= 64, epochs=100, validation_split=0.2)

        

# %%
# save best hard triplet weights
network_train.save_weights('/home/drajani/Downloads/ROB 590-20230406T190632Z-001-20230406T191441Z-001/ROB 590-20230406T190632Z-001/ROB 590/ZUHRN_AUDIO/R12/Weights_R12_HARD/')

# %%
from matplotlib import pyplot as plt

plt.plot(loss_ht.history['loss'])
plt.plot(loss_ht.history['val_loss'])
plt.title('Losses (Hard triplets)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'], loc='upper right')
plt.show()

# %%
"""
## Evaluation Metrics
"""

# %%
# Testing on an trained network
img_rows, img_cols = 128, 22


probs,yprob = compute_probs(network,new_y_tensor,new_y_tensor_L)
fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
draw_roc(fpr, tpr,thresholds)
draw_interdist(network,8800)

# %%
nb_classes = len(TimeList)
for i in range(nb_classes):
  print("Class {0}".format(i))
  AudioDist(network,np.expand_dims(Dataset_Test[i][0,:,:,:],axis=0))
