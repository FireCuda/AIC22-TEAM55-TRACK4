# AIC22-TEAM55-TRACK4

# Download Data and Weights

https://drive.google.com/drive/folders/1bIIqR-5ZD8-7Rfcp7T_YJVnh4a-BDCS9?usp=sharing

cd yolor/

Download: "yolor_p6.pt" for pretrain model, "best.pt" is our weight

File "tray_detect_mosaicless_2303.pt" is the weight for ROI detect, we just take label tray from video testA, not special so you can take this weight to detect.

Extract "out.zip" for classify model

Extract "data.zip" into folder "/binary_classification" for train classify

Extract "data_train_yolor.zip" for train yolor

# Installation

pip install -r requirements.txt

git clone https://github.com/JunnYu/mish-cuda

cd mish-cuda

python setup.py build install

cd ..

# Training

## YOLOR:

python train.py

#after training the weight will save in runs/train/exp/weights/best.pt

## VIT:

cd binary_classification 

python train_example_deit.py

# Run results

python detect.py --weights best.pt


