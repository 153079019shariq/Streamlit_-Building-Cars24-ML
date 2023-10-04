# import module
import streamlit as st
 
# Title 
st.title("this is the title!!!")


# Header
st.header("This is a header")
 




# checkbox
# check if the checkbox is checked
# title of the checkbox is 'Show/Hide'
if st.checkbox("Show/Hide"):
  # display the text if the checkbox returns True value
  st.text("Showing the widget")



# radio button
# first argument is the title of the radio button
# second argument is the options for the ratio button
status = st.radio("Select Gender: ", ('Male', 'Female'))
 
# conditional statement to print
# Male if male is selected else print female
# show the result using the success function
if (status == 'Male'):
    st.success("Male")
else:
    st.success("Female")


# Create a button, that when clicked, shows a text
if(st.button("Click me")):
    st.text("button click")





num = st.number_input('Insert a number',0)

check = st.text_input("Enter the text")
 
