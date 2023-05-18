# Martian-Terrain-Classification-An-Auditory-Approach


## Project Structure
There are basically two files which contain all the datasets (and where according to the code) you will store all the intermediate audio clips that\ 
you create while audio processing for classification:
- ZUHRN_AUDIO : contains all the trajectory and audio data that we get from Zuhrn's [1] dataset which we obtain from http://deepterrain.cs.uni-freiburg.de/. We use this dataset as our baseline and an additional dataset to test our Blind Source Separation (BSS) and Siamese Neural Network architecture.
- Audio files : contains all the data and experiments done for Martian audio data from the perseverance rover.


----
Below First we show the project tree for data extraction and storing purposes according to the code for ZUHRN_AUDIO.\
Now before we dive into this structure I will first provide some pointers on this structure:
* Whenever I mention about R01, R12 or any R## means it is a trajectory and is numbered according to Zuhrn's dataset.
* In the table below:
  * Each trajectory is divided into its respective terrains, for instance, consider R02, now this is trajectory no. 02 which has two terrains constituting it i.e.\ GRASS (data given by GRASS_R02) and GRAVEL (data given by GRAVEL_R02).

```
ZUHRN_AUDIO
├── R01
├── R02
├── R03
│   ├── GRASS_R03
│   ├── GRAVEL_R03
│   └── PL_R03
├── R04
│   ├── ASPHALT_R04
│   ├── COBBLE_R04
│   ├── GRASS_R04
│   └── GRAVEL_R04
└── R12
```


  * Now GRASS has its own separate recordings that I had manually scraped from the audio-visual data from [1]. Hence, you can see three clips **'R02_grass 0.wav', 'R02_grass 1.wav' & 'R02_grass 2.wav'**. These clips were at different instances in the video dataset hence overall we have 3 clips of grass according to its presence in the video for trajectory 02. Now since we need one continuous audio clip for processing in our project's core algorithm, hence I joined this grass into one single GRASS audio clip which is given by 'GRASS_joined_R02.wav'. Now the same has been done for GRAVEL terrain.
  * In our project we have two types of triplet sampling due to the presence of siamese neural network: One is RANDOM sampling and other is HARD sampling which I created (for efficient triplet sampling and to reduce innate undue weightage which might be given to a terrain class due to random sampling). Now if you see the R02 file tree again, wherever there are stored weights they are divided into RANDOM and HARD. These weights are from these sampling method which I just mentioned. In code we first use random sampling then train the neural net then save its weights. Subsequently, we use hard sampling then again train  the neural net from scratch (this is done to show how effective my hard sampling method is over the conventional random sampling.)
  * Now again if you see R02 file, the data handling/intermediate audio clip storage/complete data storage for Baseline apporach is done in sub-directory **'R02_Testing'.** Here, files with **'__Final.wav_'** in their name are basically the ones we get after data augemntationa nd audio duration equalization in the code. The sub-sub-directories with **'__chunks_'** in their name basically store the small chunks of respective '__Final.wav_' audio files. This chopping of audio files into small chunks is done for creating a discrete training and testing dataset for training neural net, since the neural net doesnt work on continuous dataset. The weights from the Baseline training are stored in directory **'Weights_R02'**.
  * The results from Blind Source Separation (BSS) method are store in directory **'ORIGINAL_R02'**. In this directory, **'audio.wav'** is the original mixed audio fo the whole trajectory that we get directly from Zuhrn [1]'s dataset. In the code when we segment this audio file into terrain-specific audio, we store them in the directory **'splitting audio'** (here, in R02, files having **'_Seg_Signal '** in their name are those segmented terrain audio). Then same as creating chunks(as described in last pointer) we split all of these terrain-specific audio signals into small clips for training and store them in respective sub-directories (the ones having **'_Splitted_seg_Signal__'** in their name). The weights after BSS are stored in their weights file in the **'ORIGINAL_R02'** directory itself.
 
  
 <strong> ** Now the same process, as described in above pointers, is repeated for R01 and R12 trajectories throughout the project. ** </strong>.
  




  




Due to time and space limitations in the report I was able to test my code on R01,R02 & R12 trajectories. Hence, I have elaborated on these trajectories in the ensuing table.



<table id="example-table">
  <tbody>
    <tr>
      <td>R01 (Zuhrn dataset Trajectory 01)</td>
      <td>R02 (Zuhrn dataset Trajectory 02)</td>
      <td>R12 (Zuhrn dataset Trajectory 12)</td>
    </tr>
    <tr> 
<td>

```
R01
├── ASPHALT_R01
│   └── R01_asphalt 0.wav
├── COBBLE_R01
│   ├── COBBLE_joined_R01.wav
│   ├── R01_cobble 0.wav
│   └── R01_cobble 1.wav
├── GRASS_R01
│   └── R01_grass 0.wav
├── GRAVEL_R01
│   ├── GRAVEL_joined_R01.wav
│   ├── R01_gravel 0.wav
│   ├── R01_gravel 1.wav
│   └── R01_gravel 2.wav
├── ORIGINAL_R01 (Weights: https://drive.google.com/file/d/1uF5BZdE4yw4zzCgVWOqu9_9WkJYbt3Hr/view)
│   ├── audio.wav
│   ├── splitting audio
│   │   ├── Seg_Signal 0.wav
│   │   ├── Seg_Signal 1.wav
│   │   ├── Seg_Signal 2.wav
│   │   ├── Seg_Signal 3.wav
│   │   ├── Splitted_seg_signal_0
│   │   ├── Splitted_seg_signal_1
│   │   ├── Splitted_seg_signal_2
│   │   └── Splitted_seg_signal_3
│   ├── weights_HARD
│   │   └── checkpoint
│   └── weights_RANDOM
│       └── checkpoint
├── R01_Testing
│   ├── asphalt_chunks
│   ├── asphalt_Final.wav
│   ├── cobbelstone_chunks
│   ├── cobbelstone_Final.wav
│   ├── grass_chunks
│   ├── grass_Final.wav
│   ├── gravel_chunks
│   └── gravel_Final.wav
└── Weights_R01 (https://drive.google.com/file/d/1TAROUfxI_QULNwoj0GYepHsUtyC7wkVx/view)
    ├── Weights_R01_HARD
    │   └── checkpoint
    └── Weights_R01_RANDOM
        └── checkpoint
```   
</td>
<td>

```
R02
├── GRASS_R02
│   ├── GRASS_joined_R02.wav
│   ├── R02_grass 0.wav
│   ├── R02_grass 1.wav
│   └── R02_grass 2.wav
├── GRAVEL_R02
│   ├── GRAVEL_joined_R02.wav
│   ├── R02_gravel 0.wav
│   ├── R02_gravel 1.wav
│   ├── R02_gravel 2.wav
│   ├── R02_gravel 3.wav
│   └── R02_gravel 4.wav
├── ORIGINAL_R02 (Weights: https://drive.google.com/file/d/1Q43YMrecRVWDNgNWbx7Wo5re9NOePO29/view)
│   ├── audio.wav
│   ├── splitting audio
│   │   ├── Seg_Signal 0.wav
│   │   ├── Seg_Signal 1.wav
│   │   ├── Splitted_seg_signal_0
│   │   └── Splitted_seg_signal_1
│   ├── weights_HARD
│   │   └── checkpoint
│   └── weights_RANDOM
│       └── checkpoint
├── R02_Testing
│   ├── grass_chunks
│   ├── grass_Final.wav
│   ├── gravel_chunks
│   └── gravel_Final.wav
└── Weights_R02 (https://drive.google.com/file/d/1B5ia0OWTH1eR1DCBA-znE9cYDX4j_50c/view)
    ├── Weights_R02_HARD
    │   └── checkpoint
    └── Weights_R02_RANDOM
        └── checkpoint

```
</td><td>

```
R12
├── ASPHALT_R12
│   ├── ASPHALT_joined_R12.wav
│   ├── R12_asphalt 0.wav
│   └── R12_asphalt 1.wav
├── COBBLE_R12
│   └── R12_cobble 0.wav
├── GRASS_R12
│   ├── GRASS_joined_R12.wav
│   ├── R12_grass 0.wav
│   └── R12_grass 1.wav
├── GRAVEL_R12
│   ├── GRAVEL_joined_R12.wav
│   ├── R12_gravel 0.wav
│   ├── R12_gravel 1.wav
│   ├── R12_gravel 2.wav
│   ├── R12_gravel 3.wav
│   └── R12_gravel 4.wav
├── ORIGINAL_R12 (Weights: https://drive.google.com/file/d/1cwdiK44DHV1BUwTno4nfm6zBiDyZFX1w/view)
│   ├── audio.wav
│   ├── splitting audio
│   │   ├── Seg_Signal 0.wav
│   │   ├── Seg_Signal 1.wav
│   │   ├── Seg_Signal 2.wav
│   │   ├── Seg_Signal 3.wav
│   │   ├── Seg_Signal 4.wav
│   │   ├── Splitted_seg_signal_0
│   │   ├── Splitted_seg_signal_1
│   │   ├── Splitted_seg_signal_2
│   │   ├── Splitted_seg_signal_3
│   │   └── Splitted_seg_signal_4
│   ├── weights_HARD
│   │   └── checkpoint
│   └── weights_RANDOM
│       └── checkpoint
├── PL_R12
│   └── R12_PL 0.wav
├── R12_Testing
│   ├── asphalt_chunks
│   ├── asphalt_Final.wav
│   ├── cobbelstone_chunks
│   ├── cobbelstone_Final.wav
│   ├── grass_chunks
│   ├── grass_Final.wav
│   ├── gravel_chunks
│   ├── gravel_Final.wav
│   ├── PL_chunks
│   └── PL_Final.wav
└── Weights_R12 (https://drive.google.com/file/d/1ezOENg325jbmLQsxoX0qP1nTEr-92V94/view)
    ├── Weights_R12_HARD
    │   └── checkpoint
    └── Weights_R12_RANDOM
        └── checkpoint


```
</td>
    </tr>
  </tbody>
</table>

----
Now for another dataset file **Audio files** the structure is as follows:

```
Audio files

├── Rover_Audio.wav
|
└── new_Splitting_20sec__4_Rover_Audio

    ├── Seg_Signal 0.wav

    ├── Seg_Signal 1.wav

    ├── Seg_Signal 2.wav

    ├── Seg_Signal 3.wav

    ├── Splitted_seg_signal_0

    ├── Splitted_seg_signal_1

    ├── Splitted_seg_signal_2

    ├── Splitted_seg_signal_3

    ├── WEIGHTS_Martian audio_random (https://drive.google.com/file/d/1ujLW_I5GFUhtcW_eVZw-sk6zIW4XLQae/view)

        └── checkpoint

    └── WEIGHTS_Martian audio_hard (https://drive.google.com/file/d/1Bu-MW6mjDjbwrPyS3NIQim73gmUt5_dW/view)

        └── checkpoint
```

In this file all the data handling regarding the Martian Audio is done.
* Main 16 min. audio clip, which is mixed and raw, that we directly get from the rover is **'Rover_Audio.wav'**.
* Using BSS method this audio clip is segmented into 4 terrain specific audio clips given by file having thier names **'Seg_Signal #.wav'** in this format.
* Now each of these terrain specific audio is chopped into small chunks of audio clips (just as done in Zuhrn dataset data) and stored in corresponding sub-directories with names **'Splitted_seg_signal_#'**.
* The hard and random triplet training weights are stored in corresponding weights file in the main directory **'Audio files'**.

## References
[1] Jannik Zürn, Wolfram Burgard, Abhinav Valada
    Self-Supervised Visual Terrain Classification from Unsupervised Acoustic Feature Learning
    IEEE Transactions on Robotics (T-RO), vol. 37, no. 2, pp. 466-481, 2019. 
