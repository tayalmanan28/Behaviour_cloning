# Behaviour Cloning

## Setup

To install Anaconda follow the instructions in the following webpage:  
https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

Create a conda environment for the PyBullet tutorial:  
```
$ conda create --name behaviour_clone  
```
Switch to the newly create environment (you will notice the name of the environment on the command line in the extreme left):  
```
$ conda activate behaviour_clone  
```

Once in the desired environment install the following packages:  
```
$ conda install nb_conda_kernels  
```

Install OpenAI Gym (while in the environment):  
```
$ pip install gym==0.18.3 
```

Install Matplotlib (while in the environment):
```
$ conda install matplotlib
```


## Videos
* [Manual Expert Training](https://www.youtube.com/watch?v=XB2HAFfluEM)

* [Final Test Video](https://www.youtube.com/watch?v=ZWKFebktNMs)
