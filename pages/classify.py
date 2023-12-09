import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import os
import cv2
import glob
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
import streamlit_home
import sqlite3
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#from streamlit.media_file_manager import MediaFileStorageError


#import main , tulis main.add
#from main import func(), terus tulis fx apa = add()
#import main as m, tulis m.add()

#st.set_page_config(initial_sidebar_state="collapsed")
#st.markdown("""<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>""", unsafe_allow_html=True)
#st.markdown("<style>div.css-fblp2m {visibility: hidden;}</style>", unsafe_allow_html=True)
#st.set_option('server.sidebar.hide_on_streamlit_commands', True)
st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)
hide_sidebar_button_style = """
        <style>
        #sidebar-fblp2m {
            display: none;
        }
        </style>
        """
st.markdown(hide_sidebar_button_style, unsafe_allow_html=True)

with st.sidebar:
    #with st.echo():
        st.header("Notes")
        st.write("\n")
        st.write("Hello, I am Muhammad Akhram bin Suhaimy and this is my final year project which is cartoon character classification. Feel free to try it!")

def check_null(s):
    if s is None or s.strip() == '':
        return True
    else:
        return False
    
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\USER\vs_workspace\image_classification\FINALWEIGHTS.h5")
    return model

model = load_model()

def preprocess_image(im): 
    #new_im = tf.keras.utils.load_img(im, target_size = (224,224))
    #x = tf.keras.utils.img_to_array(new_im)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

        # Create an ImageDataGenerator object with zoom range
    datagen = ImageDataGenerator(zoom_range=[0.1, 1])

    # Load and preprocess your image
    image = tf.keras.preprocessing.image.load_img(im, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image.reshape((1,) + image.shape)  # Reshape to (1, height, width, channels)

    # Generate zoomed images
    zoomed_images = datagen.flow(image, batch_size=1)

    # Extract the zoomed image
    zoomed_image = next(zoomed_images)
    zoomed_image = zoomed_image.reshape((zoomed_image.shape[1:]))  # Reshape to (height, width, channels)
    zoomed_image = tf.keras.preprocessing.image.array_to_img(zoomed_image)
    
    array_image = tf.keras.utils.img_to_array(zoomed_image)
    array_imagee = np.expand_dims(array_image, axis=0)
    array_imagee = preprocess_input(array_imagee)

    return array_imagee


def predict(x):
    predss = model.predict(np.array(x))
    class_labels = ['all','didi','didijojo','didinana','jojo','jojonana','nana']
    pred = np.argmax(predss,axis = -1)
    label = class_labels[pred[0]]
    return label

def output(img):
    #if img is not None :
     #   real = preprocess_image(img)
        prediction = predict(img)
        st.write("the character in the image is : " + prediction)

def output_video(img):
    #if img is not None :
     #   real = preprocess_image(img)
        
        prediction = predict(img)
        #statement = st.write("the character in the image is most probably : " + prediction)
        return prediction

currentframe=0
didi=0
jojo=0
nana=0
didijojo=0
didinana=0
jojonana=0
all=0
fps=0
frames=0
duration = 0


st.header("Classify")
st.set_option("deprecation.showfileUploaderEncoding", False)

#st.write(st.session_state.username)
#con_first =sqlite3.connect("data_fir.db", check_same_thread=False)
#con_second =sqlite3.connect("data_second.db", check_same_thread=False)

st.success("Login successful!")
#streamlit_home.create_usertable_u()
#st.success(streamlit_home.user_password)
#st.success(streamlit_home.user_name)
#st.success("logged in as "+str(streamlit_home.select_user_first())+" "+str(streamlit_home.select_user_second()) )
choices = st.radio("Pick one :",('classify character from image', 'classify character from video'))

MAX_FILE_SIZE = 200 * 1024 * 1024


if choices =="classify character from image":
    
    image = st.file_uploader("Load any image here : ",type = ["jpg","jpip","png"])
    st.write(frames)
    #enter_image = st.button("show image")
    #if image is not None :
    #    real = preprocess_image(image)
    #if enter_image:
    #    if image is not None :
    #        st.image(image,"THE CHARACTER", channels = "BGR")
    #    else:
    #        st.warning("please upload an image")

    col1,col2 = st.columns([15,2])
    with col2:
        predict_image_button = st.button("predict")
    with col1:
        logout_button = st.button("logout")


    if logout_button:
        #del st.session_state.user
        #st.session_state.user = ""
        #del st.session_state.filename 
        #con_first =sqlite3.connect("data_first.db", check_same_thread=False)
        #streamlit_home.delete_user_data_first()
        #con_first.close()
        #con_second =sqlite3.connect("data_second.db", check_same_thread=False)
        #streamlit_home.delete_user_data_second()
        #con_second.close()
        #streamlit_home.user_name = ""
        switch_page("streamlit_home")
        

    if predict_image_button:

        try:
            if image is not None :
                real = preprocess_image(image)
                st.image(image,"THE CHARACTER", channels = "BGR")
                output(real)
            else:
                st.warning("please upload an image")
        except Exception as e:
            st.warning(f"Error : {e}. Please upload other image")

elif choices =="classify character from video":

    #character_name = st.text_input("name of character")
    uploaded_video = st.file_uploader("Load any video here : ",type = ["mp4","mov"])
    
    #st.session_state.filename = character_name

    #st.success(st.session_state.filename) 
    #if uploaded_video is not None:
     #   vid= uploaded_video.name
      #  cap = cv2.VideoCapture(vid)
       # fps = cap.get(cv2.CAP_PROP_FPS)
        #frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #duration = frames / fps
        #cap.release()

    column1,column2 = st.columns([15,2])
    with column1:
        logout_button = st.button("logout")
    with column2:
        predict_video_button = st.button("predict")

    if logout_button:
        
        #del st.session_state.user
        #streamlit_home.user_name = ""
        #st.session_state.filename = ""
        #con_first =sqlite3.connect("data_fir.db", check_same_thread=False)
        #streamlit_home.delete_user_data_first()
        #con_first.close()
        #con_second =sqlite3.connect("data_second.db", check_same_thread=False)
        #streamlit_home.delete_user_data_second()
        #con_second.close()
        switch_page("streamlit_home")

    if predict_video_button:

        try:
            if uploaded_video is not None :
                        
                #if len(uploaded_video.getvalue())>MAX_FILE_SIZE:
                #    st.warning("File size exceed maximum limit. Please upload file with maximum size of 200mb only.")
                        vid = uploaded_video.name
                        st.video(vid)
                        vidcap = cv2.VideoCapture(vid)
                        st.write("Classifying characters in the video...")
                        while(True):
                            success, frame = vidcap.read()
                                
                            if success :
                                filename = "image.jpg"
                                cv2.imwrite(filename, frame)
                                real = preprocess_image(filename)
                                pred = output_video(real)
                                if str(pred)=="didi":
                                    didi+=1
                                    #st.write(" "+str(pred))
                                elif str(pred)=="jojo":
                                    jojo+=1
                                    #st.write(" "+str(pred))
                                elif str(pred)=="nana":
                                    nana+=1
                                    #st.write(" "+str(pred))
                                elif str(pred)=="didijojo":
                                    didi+=1
                                    jojo+=1
                                    #st.write(" "+str(pred))
                                elif str(pred)=="didinana":
                                    didi+=1
                                    nana+=1
                                    #st.write(" "+str(pred))
                                elif str(pred)=="jojonana":
                                    jojo+=1
                                    nana+=1
                                    #st.write(" "+str(pred))
                                elif str(pred)=="all":
                                    didi+=1
                                    jojo+=1
                                    nana+=1
                                    #st.write(" "+str(pred))
                            else:
                                break

                            if cv2.waitKey(1) & 0xFF== ord("q"):
                                break
                        vidcap.release()
                        cv2.destroyAllWindows()

                        cap = cv2.VideoCapture(vid)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        duration = frames / fps
                        cap.release()
                        # Display the duration of the video file
                        st.write(f'Duration: {duration:.2f} seconds , Frames: {frames}')



                        
                        

                        st.write("total frames: "+str(frames))
                        if didi!=0:
                            didi_duration = didi/frames*duration
                            st.write("didi :"+str(didi))
                            st.write(didi_duration)
                        else:
                            didi_duration =0
                            st.write("didi :"+str(didi))
                            st.write(didi_duration)
                        if jojo!=0:
                            jojo_duration = jojo/frames*duration
                            st.write("jojo :"+str(jojo))
                            st.write(jojo_duration)
                        else:
                            jojo_duration =0
                            st.write("jojo :"+str(jojo))
                            st.write(jojo_duration)
                        if nana!=0:
                            nana_duration = nana/frames*duration
                            st.write("nana :"+str(nana))
                            st.write(nana_duration)
                        else:
                            nana_duration =0
                            st.write("nana :"+str(nana))
                            st.write(nana_duration)
                        st.write(f"didi appearance: {didi_duration:.2f}"+"seconds")
                        st.write(f"jojo appearance: {jojo_duration:.2f}"+"seconds")
                        st.write(f"nana appearance: {nana_duration:.2f}"+"seconds")
                        didi=0
                        jojo=0
                        nana=0
                        didi_duration =0
                        jojo_duration =0
                        nana_duration =0
            else:
                st.warning("Please upload a video")
        except Exception as e:
            st.warning(f"Error : {e}. Please choose other video")
            
        
        
        
        
    






        
