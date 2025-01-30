import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import tempfile
import base64
from pose_estimation import detect_pose
from pose_estimation_video import process_video

st.title("Human Pose Estimation Web App")

option = st.sidebar.selectbox("Select Input Type", ("Image", "Video"))

def get_image_download_link(img, filename="pose_estimated.png", text="Download Processed Image"):
    _, img_encoded = cv2.imencode(".png", img)
    b64 = base64.b64encode(img_encoded).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        if st.button("Show Pose Estimation"):
            processed_image = detect_pose(image)
            st.image(processed_image, caption="Pose Estimated Image", use_container_width=True)
            st.markdown(get_image_download_link(processed_image), unsafe_allow_html=True)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        if st.button("Show Pose Estimation"):
            try:
                output_video_path = process_video(temp_path)
                st.video(output_video_path)

                with open(output_video_path, "rb") as file:
                    video_bytes = file.read()
                    b64 = base64.b64encode(video_bytes).decode()
                    href = f'<a href="data:video/mp4;base64,{b64}" download="pose_estimated.mp4">Download Processed Video</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Cleanup temporary files
                os.remove(output_video_path)
                os.remove(temp_path)
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")