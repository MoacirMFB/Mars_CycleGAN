# Mars_CycleGAN

#Dataset

> Download BW and Color images from 

https://mars.nasa.gov/msl/multimedia/images/?page=0&per_page=25&order=pub_date+desc&search=&category=51%3A176&fancybox=true&url_suffix=%3Fsite%3Dmsl

> Put BW images under path named dataset/train/trainA
> Put Color images under path named dataset/train/trainB


#Installation requirements 

#Install Miniconda if not available 

  wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
  bash Miniconda3-py39_4.11.0-Linux-x86_64.sh
  source ~.bashrc
  conda update -n base -c defaults conda
  conda install anaconda-client -n base

#Create environment
  conda env create -f environment.yml 

#Additional packages required 
  conda install -c conda-forge albumentations
  conda install tqdm

#If applicable
#Install a PyTorch version that supports A30 NVIDIA 

  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install albumentations 

#Activate environment
  conda activate pytorch-CycleGAN-and-pix2pix

#Run 
  python3 train.py 
