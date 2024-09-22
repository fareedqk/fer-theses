# import libraries
import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.utils  import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
emotion_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6: 'Surprise'}
# load json and create model
json_file = open('./models/fer_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("./models/fer_model.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img
    

    # Function to perform facial expression recognition on uploaded image
    def recognize_emotions(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(image, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image

def main():
    # Setting page layout
    st.set_page_config(
        page_title="FER",
        page_icon="🎥",
        layout="wide",
        # initial_sidebar_state="expanded"
)

    # Face Analysis Application 
    st.title("🎥 Real Time Facial Expression Recognition")
    activities = ["Home", "Webcam Face Detection", "Upload Image" ,"About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown("**🎯 Let's Connect**")
    st.sidebar.markdown(
        "- [LinkedIn](https://www.linkedin.com/in/fareedcodes/)\n"
        "- [GitHub](https://github.com/fareed23/)\n"
        "- [X](https://twitter.com/fareedcodes/)\n"
        "- [Kaggle](https://kaggle.com/fareedcodes/)"
    )
    
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Start web cam and check for real time facial expressions.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time facial detection and expression recognition using web cam feed.

                 3. Facial expression recognition for an uploaded image.

                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your facial expressions")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
        

    elif choice == "Upload Image":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Recognize Facial Expressions'):
                image_with_emotions = Faceemotion.recognize_emotions(image)
                st.image(image_with_emotions, caption='Facial Expressions Recognized', use_column_width=True)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time facial expression recognition application using Streamlit Framework, Opencv, Tensorflow and Keras library.</h4>
                                    <h4 style="color:white;text-align:center;">
                                    If there are any recommendations or improvements, you can utilize the contact links given.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
