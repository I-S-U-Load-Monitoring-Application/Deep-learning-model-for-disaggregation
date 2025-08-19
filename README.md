UKDALE Seq2Point NILM

This repository contains an implementation of a Seq2Point deep learning model for Non-Intrusive Load Monitoring (NILM) using the UKDALE dataset.
The goal is to disaggregate household energy consumption and estimate the power usage of individual appliances from aggregate mains data.

📖 Project Overview

Task: Energy disaggregation (predicting appliance-level consumption from total mains readings).

Model: Convolutional Neural Network (CNN) based Seq2Point architecture.

Dataset: UKDALE
 (UK Domestic Appliance-Level Electricity) dataset.

Framework: TensorFlow / Keras (implemented in Jupyter Notebook).

Seq2Point reformulates NILM as a supervised learning problem by sliding a window over aggregate mains data and predicting the midpoint appliance power value.

⚙️ Requirements

Install dependencies before running the notebook:

pip install tensorflow numpy pandas matplotlib scikit-learn


If working directly with NILM datasets:

pip install nilmtk h5py

📂 Dataset

Download the UKDALE dataset from the official source:
UKDALE dataset

Preprocess the dataset into a format suitable for training/testing.

Update dataset paths inside the notebook.

🚀 Usage

Clone this repository:

git clone https://github.com/I-S-U-Load-Monitoring-Application/Deep-learning-model-for-disaggregation.git
cd UKDALE-seq2point


Open Jupyter Notebook:

jupyter notebook


Run the notebook:

UKDALE seq2point.ipynb

Train the model on appliance-level data.

Evaluate disaggregation performance.

📊 Output

Trained model weights.

Disaggregated appliance power consumption.

Plots comparing true vs predicted appliance consumption.

Metrics such as MAE, RMSE, and accuracy.

🔬 References

Kelly, J., & Knottenbelt, W. (2015).
The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes.
Link

Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C. (2018).
Sequence-to-point learning with neural networks for non-intrusive load monitoring.
Paper

📌 Notes

The notebook can be adapted for other NILM datasets such as REFIT or REDD.

Training on full datasets requires significant compute (GPU recommended).

Model can be exported to .h5 or .tflite for deployment in edge/IoT devices.
