# UCNet_Steganalysis

This is the PyTorch implementation of the manuscript "Universal Deep Network for Steganalysis of Color Image based on Channel Representation". 

# Requirements:
CUDA (10.2)
cuDNN (7.4.1)
python (3.6.9)

# Use
"UCNet_model.py" is the main program of the model 

"High-pass filters" contains the 30 SRM filters 


Example: 
If you want to detect CMDC-HILL steganography method at 0.4 bpc (on GPU #1), you can enter following command:

"python3 UCNet_Spatial.py -alg CMDC-HILL -rate 0.4 -g 1"


# Note
If you have any question, please contact me. (revere.wei@outlook.com)
