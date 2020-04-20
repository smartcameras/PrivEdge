# PrivEdge

This is the official repository of [PrivEdge: From Local to Distributed Private Training and Prediction](https://arxiv.org/pdf/2004.05574.pdf), a work published in the EEE Transactions on Information Forensics and Security (TIFS), April, 2020.<br>

## Setup

1. Download source code from GitHub
   ```
    git clone https://github.com/smartcameras/PrivEdge.git 
   ```
2. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name PrivEdge python=2
   ```
3. Activate conda environment
   ```
    source activate PrivEdge
   ```
4. Install requirements
   ```
    pip install -r requirements.txt
   ```
   Install keras_contrib
   ```
    git clone https://www.github.com/keras-team/keras-contrib.git
    cd keras-contrib
    python setup.py install
   ```

## Description

PrivEdge is a technique for privacy-preserving MLaaS that safeguards the privacy of users who provide their data for training, as well as users who use the prediction service. We decompose an N-class classifier into N one-class classifiers. With PrivEdge, each user independently uses their private data to locally train a one-class reconstructive adversarial network (RAN) that succinctly represents their training data. The training phases of the one-class RANs were implemented in Python with the publicly available Keras library.
For private prediction, we assume that a non-colluding regulator is available and use the 2-server model of multi-party computation (2PC). We used the ABY library for secure 2PC (i.e. additive secret-sharing and Garbled circuit) with 128-bit security parameter and SIMD circuits.


### Data distribution

We model each user as a distinct class:

1. Go to Dataset directory 
```
cd Dataset
```
2. Create/Download your dataset:
```
wget https://www.kaggle.com/olgabelitskaya/classification-of-handwritten-letters/version/9#LetterColorImages_123.h5
```
3. Split and save the N-class dataset to N set for N users:
```
python Distribute_data.py
```

### Local training
Each user train locally a one-class RAN, which is composed of a reconstructor and a discriminator, on their private data:
```
python run_train.py
```
The trained RAN will be save in the saved_model directory. Some visualization of results also will be saved in the images directory.

### Prediction
It includes the private reconstruction of each one-class classifier followed by dissimilarity based prediction. We do the accuracy experiments in the python while timing the secure protocols of our prediction using ABY (coming soon):
```
python test.py
```

## Authors
* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Adrià Gascón](mailto:agascon@turing.ac.uk)
* [Hamed Haddadi](mailto:h.haddadi@imperial.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)


## References
If you use our code, please cite the following paper:

      @article{shamsabadi2020privedge,
        title = {PrivEdge: From Local to Distributed Private Training and Prediction},
        author = {Shamsabadi, Ali Shahin and Gascón, Adrià and Haddadi, Hamed and Cavallaro, Andrea},
        journal = {IEEE Transactions on Information Forensics and Security (TIFS)},
        year = {2020},
        month = April
      }
      
## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).
