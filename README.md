# UCNet_Steganalysis

This is the PyTorch implementation of the paper "Universal Deep Network for Steganalysis of Color Image based on Channel Representation", TIFS 2022. 

# Requirements:
CUDA (10.2)
cuDNN (7.4.1)
python (3.6.9)

# Use
"UCNet_Spatial.py" and "UCNet_JPEG.py" are the main program in spatial and JPEG domain, respectively. 

"High-pass filters" contains the 30 SRM filters. 

"J-UNIWARD-pretrain-parameters.pt" is pretrain parameter under a small ImageNet (training/validation:190,000/8,820 images) with J-UNIWARD steganography. 

Example: 

If you want to detect CMDC-HILL steganography method at 0.4 bpc (on GPU #1), you can enter following command:

"python3 UCNet_Spatial.py -alg CMDC-HILL -rate 0.4 -g 1"


# Note
If you have any question, please contact me. (weikk5@mail2.sysu.edu.cn)
