# Cotton Weed Detection Challenge

Accurate weed detection is a critical component of precision agriculture, enabling targeted intervention and reduced reliance on herbicides. In this project, we participate in the 3LC Cotton Weed Detection Challenge, which focuses on detecting and classifying three types of weeds: Carpetweed, Morning Glory, and Palmer Amaranth in cotton field images. The challenge simulates re-alistic agricultural conditions with imperfect labels and strict model size constraints. Our work benchmarks the YOLOv8n object detection model and emphasizes adata-centric AI workflow rather than increasing model complexity. Using the 3LC platform, we conduct iterative error analysis, dataset cleaning, and retraining through a train-fix-retrain loop. Our results show that improving annotation quality and correcting dataset inconsistencies can significantly enhance detection performance, demonstrating the importance of data quality in real-world computer vision systems.

The datasets used for training and evaluation can be found [on the competiton data page](https://www.kaggle.com/competitions/the-3lc-cotton-weed-detection-challenge/data)

# Connecting OSC to 3LC
`ssh -L 5015:127.0.0.1:5015 <USERNAME>@pitzer-login<GET_NUMBER_FROM_OSC>.hpc.osc.edu`

`cd /fs/scratch/PAS3162/smith.15485/cotton_weed_competition_dataset`

`source cotton-weed-env/bin/activate`

`3lc service`

# Training Model
Modify the training parameters in `train.py` and ensure the image dataset is in the proper folder.
`python3 train.py`

# Generating Predictions
Modify the model .pt file name in `predict.py` and ensure the test image dataset is in the proper folder.
`python3 predict.py`

# Evaluating Predictions
Prediction files are stored in `result.csv`
`python3 evaluate_validation.py`