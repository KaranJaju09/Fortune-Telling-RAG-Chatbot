# Horoscope Chatbot

This project is a simple chatbot that provides horoscope readings for different zodiac signs. It uses a combination of LangChain, Streamlit, and a Hugging Face model to generate responses.

## Features

- Select your zodiac sign.
- Ask questions about your future.
- Get personalized horoscope readings.

## How it Works

The chatbot uses a Retrieval-Augmented Generation (RAG) model. It retrieves relevant information from a knowledge base of horoscope texts based on your zodiac sign and question. Then, it uses a large language model to generate a response based on the retrieved information.

The project is structured as follows:

- `main.py`: Contains the core chatbot logic, including the RAG model and the `ChatBot` class.
- `run_streamlit.py`: Creates a simple web interface for the chatbot using Streamlit.
- `data/`: Contains the horoscope texts for each zodiac sign.
- `faiss_index/`: Stores the FAISS index for efficient similarity search.
- `requirements.txt`: Lists the required Python packages.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/horoscope-chatbot.git
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file:**
   Create a `.env` file in the root directory of the project and add the following line:
   ```
   HUGGINGFACEHUB_API_TOKEN='your-hugging-face-api-token'
   ```
   Replace `your-hugging-face-api-token` with your actual Hugging Face API token.

## Usage

To run the chatbot, execute the following command:
```bash
streamlit run run_streamlit.py
```
This will start a local web server and open the chatbot interface in your browser.