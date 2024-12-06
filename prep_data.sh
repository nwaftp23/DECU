#!/bin/bash

cd imagenet_data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
tar -xvf ILSVRC2012_img_train.tar
mkdir ./data_train/data
rm ILSVRC2012_img_train.tar
for f in *.tar; do tar -xvf "$f" --one-top-level; done
rm *.tar
mv n* ./data_train/data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar
mkdir ./data_val/data
mv *.JPEG ./data_val/data
rm ILSVRC2012_img_val.tar
cd .. 
# mkdir ./pretrained_models
# mkdir ./pretrained_models/diffusion
# mkdir ./pretrained_models/vae
# wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
# wget https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
# unzip vq-f4.zip -d ./pretrained_models/vae
# rm vq-f4.zip
# mv model.ckpt ./pretrained_models/diffusion
# python38 create_val_dirs.py
