# Human Pose Estimation using OpenPose  

##  Overview  
This project implements **human pose estimation** using **OpenPose** and provides a **Streamlit-based web app** for processing images and videos. Users can upload files to visualize detected keypoints and download processed results.  

##  Features  
- Detects human body keypoints from images & videos
- Web-based interface using **Streamlit**  
- Supports **image & video processing**  
- Allows downloading processed outputs  

##  Technologies Used  
- **Python**  
- **OpenCV**  
- **Streamlit**  
- **TensorFlow/OpenPose**  
- **NumPy**  

## Installation  
To set up the project, run:  
```bash
git clone https://github.com/honey1088/Human_Pose_Estimation
cd human-pose-estimation
pip install -r requirements.txt

##  Download OpenPose Models
The `models/` folder is not included in this repository. Please download the required OpenPose models from:
[OpenPose GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.sh).

##  Usage
Run: streamlit run app.py
```
