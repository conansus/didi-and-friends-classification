import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from PIL import Image
import sqlite3
import pandas as pd
 

#TO GET RID OF SIDEBAR
st.set_page_config(page_title  = "Home", initial_sidebar_state="collapsed")
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


#DATABASE
con =sqlite3.connect("database2.db", check_same_thread=False)
c = con.cursor()

def create_usertable():
    con =sqlite3.connect("database2.db", check_same_thread=False)
    c = con.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS userstable_new(firstname TEXT, secondname TEXT, username TEXT, password TEXT)")
        
def add_userdata(firstfullname, secondfullname, username, password):
    con =sqlite3.connect("database2.db", check_same_thread=False)
    c = con.cursor()
    c.execute("INSERT INTO userstable_new(firstname, secondname, username,password) VALUES(?,?,?,?)", (firstfullname, secondfullname, username, password))
    con.commit()

def login_user(username, password):
    con =sqlite3.connect("database2.db", check_same_thread=False)
    c = con.cursor()
    c.execute("SELECT * FROM userstable_new WHERE username = ? AND password = ?", (username, password))
    data = c.fetchall()
    return data

def view_all_users():
     con =sqlite3.connect("database2.db", check_same_thread=False)
     c = con.cursor()
     c.execute("SELECT * FROM userstable_new")
     data = c.fetchall()
     return data

def delete_user_data():
    con =sqlite3.connect("database2.db", check_same_thread=False)
    c = con.cursor()
    c.execute("DELETE FROM userstable_new;")
    con.commit()

def takefirstname(user):
    con =sqlite3.connect("database2.db", check_same_thread=False)
    c = con.cursor()
    c.execute("SELECT firstname FROM userstable_new WHERE username = ?", (user,))
    data = c.fetchone()
    return str(data[0])

def takesecondname(user):
    con =sqlite3.connect("database2.db", check_same_thread=False)
    c = con.cursor()
    c.execute("SELECT secondname FROM userstable_new WHERE username = ?", (user,))
    data = c.fetchone()
    return str(data[0])
#delete_user_data()
#con.close()

def check_null(s):
    if s is None or s.strip() == '':
        return True
    else:
        return False

st.title("Welcome to our Didi and Friends classification page!")
#image = Image.open(r'C:\Users\USER\Downloads\main_page.jpg')
#st.image(image, caption='didi and friends main characters')
st.write("Here, we classify the main characters of Didi and Friends cartoon series ")
user_name = st.text_input("Username :")
user_password = st.text_input("Password :",type = "password")
login_button =  st.button("Enter")
#if "user" not in st.session_state:
#    st.session_state.user= ""
#st.session_state.user = user_name #session state boleh share between pages tkyh import
#con_first =sqlite3.connect("data_first.db", check_same_thread=False)
#create_usertable_first()
#con_second =sqlite3.connect("data_second.db", check_same_thread=False)
#create_usertable_second()


#with st.sidebar:
#    #with st.echo():
#        st.header("Notes")
#        st.write("\n")
#        st.write("Hello, I am Muhammad Akhram bin Suhaimy and this is my final year project which is cartoon character classification. Feel free to try it!")


if login_button:  
    con =sqlite3.connect("database2.db", check_same_thread=False)     
    create_usertable()
    if check_null(user_name)==False and check_null(user_password)==False:
        result = login_user(user_name, user_password) 
        con.close()
        
        if result :
        #st.success(f"logged in as {user_name}")
        #st.success("logged in as {}".format(user_name)) #f"  {} 

            first = takefirstname(user_name)
            second = takesecondname(user_name)
            #st.success(first+" "+second)
            
            #st.success(select_user_second())
            
            switch_page("classify")
        else:
            st.warning("Incorrect username/password")
    else:
        st.warning("Do not leave blank space")        

#SIGNUP BUTTON
st.write("Doesn't have an account? Sign up now ")
signup_button = st.button("signup")
if signup_button==True:
    switch_page("signup")

con =sqlite3.connect("database2.db", check_same_thread=False)
user_result = view_all_users()
db = pd.DataFrame(user_result,columns = ["firstname","secondname","username","password"])
st.dataframe(db)













