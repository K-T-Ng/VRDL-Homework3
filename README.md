# VRDL-Homework3

## Installation
We follow the installation from: https://github.com/DGMaxime/detectron2-windows </br>
### Step1: Create a conda environment
```
conda create -n HW3 python=3.7
conda activate HW3
```
### Stpe2: Install torch
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

### Step3: Install Cython and Pycocotools
```
pip install cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

### Step4: Clone this repository
clone this repository and change the directory.
```
git clone https://github.com/K-T-Ng/VRDL-Homework3.git
cd VRDL-Homework3
```
### Step5: Install Detectron2
```
git clone https://github.com/DGMaxime/detectron2-windows.git
cd detectron2-windows
pip install -e .
cd ..
```

## Folder structure
The model weights can be downloaded here: https://drive.google.com/file/d/1ACo7koEgy4CtIBGXbzZryqcbFfgdnKa9/view?usp=sharing </br>
In order to reproduce the testing result, we may need to make a folder named ```weights``` and put the weight file into this folder. </br>
After we followed the installation above and downloaded the weights, the folder structure looks like: </br>

    .
    ├──dataset
       ├──test
          ├──TCGA-50-5931-01Z-00-DX1.png
          ├──TCGA-A7-A13E-01Z-00-DX1.png
          ├──{other testing images}
       ├──train
          ├──TCGA-18-5592-01Z-00-DX1
             ├──images
                ├──TCGA-18-5592-01Z-00-DX1.png
             ├──masks
                ├──mask_0001.png
                ├──{other mask images}
          ├──{ohter training folders}
       ├──test_img_ids.json  
       ├──train.json
    ├──detectron2-windows
    ├──weights
       ├──model_final.pth
    ├──Config.py
    ├──DataLoader.py
    ├──Inference.py
    ├──PrepareDataset.py
    ├──Train.py
    ├──Trainer.py
    ├──README.md


## Reference
We follow the installation from: https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c </br>
The code that we used is: https://github.com/DGMaxime/detectron2-windows
