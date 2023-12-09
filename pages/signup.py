import streamlit as st
from streamlit_extras.switch_page_button import switch_page 
from streamlit_home import create_usertable,add_userdata
import sqlite3
import streamlit_home
#st.set_page_config(initial_sidebar_state="collapsed")
#st.markdown("""<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>""", unsafe_allow_html=True)
#st.markdown("<style>div.css-fblp2m {visibility: hidden;}</style>", unsafe_allow_html=True)
#st.set_option('server.sidebar.hide_on_streamlit_commands', True)
import streamlit as st

hide_sidebar_button_style = """
        <style>
        #sidebar-fblp2m {
            display: none;
        }
        </style>
        """
st.markdown(hide_sidebar_button_style, unsafe_allow_html=True)
st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)

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
counter=0    
st.header("Create new account here !")

st.write("Complete the details and enjoy :)")
first_fullname = st.text_input("First Name :")
second_fullname = st.text_input("Second Name :")
username = st.text_input("Create any username :")
password = st.text_input("Password : ", type = "password")
rewrite_password = st.text_input("Re-enter your password :", type = "password")


col1,col2 = st.columns([1,9])
with col1:
    back_button = st.button("back")
with col2:
    register_button = st.button("register")
def check_fullname(name):
    test = 0
    for c in name:
        if not c[0].isalpha():
            return False
    return True


def check_username(user):
    number=0
    if len(user) < 3:
        return False

    # Check that the username starts with a letter
    if not username[0].isalpha():
        return False
    
    for c in user:
        if not c.isalpha():
            return False
    return True
    
    # Check that the username contains at least one number
    #for char in user:
    #    if char.isdigit()==True:
    #        number+=1
    #if number<0:
    #    return False

    # If all checks pass, the username is valid
    return True

def check_password(passw):
    if len(passw) < 6:
        return False

    # Check that the username starts with a letter
    if not passw[0].isalpha():
        return False

    # Check that the username contains at least one number and one special character
    if not any(char.isdigit() for char in passw) or not any(char.isalpha() for char in passw) or not any(not char.isalnum() for char in passw):
        return False

    # If all checks pass, the username is valid
    return True

if back_button:
        switch_page("streamlit_home")

if register_button:
    if check_null(first_fullname)==False and check_null(second_fullname)==False and check_null(username)==False and check_null(password)==False and check_null(rewrite_password)==False:
        if check_fullname(first_fullname)==True and check_fullname(second_fullname)==True:
            if check_username(username) == True:
                if password == rewrite_password:
                    if check_password(password)==True:
                        con =sqlite3.connect("database2.db", check_same_thread=False)
                        c = con.cursor()
                        create_usertable()
                        add_userdata(first_fullname, second_fullname, username, password)
                        #con.close()
                        st.success("you have successfully created a new account !")
                        st.write("\n\n\n\n")
                        switch_page("streamlit_home")
                    else:
                        st.warning("please include at least one capital letter, special characters and numbers for the password") 
                else:
                    st.warning("make sure your password and re-enter password is same")
            else:
                st.warning('Please enter a username consisting of characters')
        else:
            st.warning('Please enter only characters')

    else:
        st.warning("please do not leave any blank space")