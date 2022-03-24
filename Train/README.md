

# This is the training codes for our AURL model.
### Descriptions
The source codes are used to train our AURL model on the kinetics 700 dataset with ![1](http://latex.codecogs.com/svg.latex?\tau=0.05).  We deploy the training on 8 Nvidia Tesla V100 GPUs, the number of training iterations is 58,500 which takes 45 hours.

You can start the codes from the execute command "start.sh". 

### Requirements
```
Python 3.6
Pytorch 1.7.1
torchvision 0.8.2
GoogleNews-vectors-negative300.bin
nltk_data

horovod
8 Nvidia Tesla V100 GPUs
```
***Pytorch and torchvision:***
	*pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html*

***GoogleNews-vectors-negative300.bin:***
	*wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz -O /workplace/word2vec/GoogleNews-vectors-negative300.bin.gz*
	*gunzip -c /workplace/word2vec/GoogleNews-vectors-negative300.bin.gz > /workplace/word2vec/GoogleNews-vectors-negative300.bin*


***nltk_data:***
	*pip3 install nltk*
	*python3 -c "import nltk; nltk.download('wordnet')"*

### Dataset
Please download related datasets:    
[Kinetics 700](https://deepmind.com/research/open-source/kinetics), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

Data structure:    
To speed up training, we offline extract all frames of training videos and save them in [lmdb](https://lmdb.readthedocs.io/en/release/) files.    
The file "/data/generate_imgslmdb.py" is used to generate lmdb files.    
The lmdb Key, class name, and lmdb path of each video are saved in "/data/kinetics_id_exist.txt".





