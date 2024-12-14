import streamlit as st
from utils import *
import cv2
import tempfile

def main(weights_path):
    st.title("Video Object Detection Web App")
    st.header("Powered by YOLOv11 from Ultralytics")

    # Uploaded file
    uploaded_file = st.file_uploader(label="Choose your video:", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.write("File has been successfully uploaded.")
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file_binary = uploaded_file.read()

        # Write temporary file
        temp_file.write(file_binary) 
    
        # Open the video file
        video_cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        # Initialize VideoWriter for saving the output video
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='mp4').name
        output_video = cv2.VideoWriter(output_video_path,
                                       cv2.VideoWriter_fourcc(*"H264"),
                                       fps,
                                       (width, height))
        
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            
            if not ret:
                break

            # Do prediction
            model = load_model(model_path=weights_path)
            results = model.predict(frame,
                                    conf=0.5)

            # Draw bounding boxes to every frames in video
            annotated_frame = drawing_bounding_boxes(frame, results)

            # Write the annotated frame to the output video
            output_video.write(annotated_frame)

            # Convert annotated frame back to RGB for displaying in Streamlit
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        video_cap.release()
        output_video.release()
        
if __name__ == '__main__':
    weights_path = "./weights/best.pt"
    main(weights_path=weights_path)
