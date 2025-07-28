"""
This module creates a simple web interface for the chatbot using Streamlit.
"""
from main import ChatBot
import streamlit as st

# Create a new instance of the ChatBot
bot = ChatBot()

# Set the page title
st.set_page_config(page_title="Fortune Telling Bot")

# Create the sidebar
with st.sidebar:
    st.title('Fortune Telling Bot')

    # Create a dropdown menu for the zodiac signs
    zodiac = st.selectbox(
        "Select your Zodiac Sign:",
        [
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
        ]
    )

def generate_response(input, zodiac):
    """
    Generates a response from the chatbot.

    Args:
        input (str): The user's input.
        zodiac (str): The user's zodiac sign.

    Returns:
        str: The chatbot's response.
    """
    result = bot.ask(input, zodiac)
    return result

# Create the chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's unveil your future"}]

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get the user's input
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input, zodiac)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
