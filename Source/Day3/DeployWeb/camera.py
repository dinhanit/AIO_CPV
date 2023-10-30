import streamlit as st
import cv2

# Title of the Streamlit app
st.title("Camera Stream with Streamlit")

# Function to capture video from the default camera (usually the webcam)
def main():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the webcam)

    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            st.warning("Warning: Could not read frame.")
            continue

        # Display the frame in the Streamlit app
        st.image(frame, channels="BGR", use_column_width=True)

    cap.release()

if __name__ == "__main__":
    main()
