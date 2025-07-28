"""
This module contains the ChatBot class, which is the core of the horoscope chatbot.
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

class ChatBot():
    """
    A chatbot that provides horoscope readings based on a user's zodiac sign.
    """
    def __init__(self):
        """
        Initializes the ChatBot by loading the environment variables, setting up the embeddings,
        loading the documents, and initializing the language model.
        """
        load_dotenv()

        # Initialize the embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.db_path = "./faiss_index"
        self.vectorstore = None

        # Load the documents and create the vector store
        self._load_documents()

        # Initialize the language model
        model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Create the prompt template
        template = """
        You are a seer. These Humans will ask you questions about their life. Use the following context to answer.
        If you don't know, just say you don't know. Be short and concise, no more than 2 sentences.

        Context: {context}
        Question: {question}
        Answer:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def _load_documents(self):
        """
        Loads the documents from the data folder, splits them into chunks, and creates a FAISS vector store.
        If the vector store already exists, it loads it from the disk.
        """
        if os.path.exists(self.db_path):
            self.vectorstore = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            return

        docs = []
        data_folder = "./data"

        # Load the documents from the data folder
        for filename in os.listdir(data_folder):
            if filename.endswith(".txt"):
                zodiac = filename.replace(".txt", "").lower()
                path = os.path.join(data_folder, filename)
                loader = TextLoader(path)
                raw_docs = loader.load()
                for doc in raw_docs:
                    doc.metadata["zodiac"] = zodiac
                docs.extend(raw_docs)

        # Split the documents into chunks
        text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=20)
        docs = text_splitter.split_documents(docs)

        # Create a preview of the chunks
        with open("all_chunks_preview.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(docs):
                f.write(f"\n\n[Chunk {i+1}] (Zodiac: {doc.metadata.get('zodiac', 'unknown')}):\n{doc.page_content}\n")

        # Create the FAISS vector store and save it to the disk
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local(self.db_path)

    def ask(self, question, zodiac):
        """
        Asks a question to the chatbot and returns the answer.

        Args:
            question (str): The question to ask.
            zodiac (str): The user's zodiac sign.

        Returns:
            str: The answer to the question.
        """
        # Retrieve the relevant documents from the vector store
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
        raw_docs = retriever.invoke(question)

        # Filter the documents by zodiac sign
        filtered_docs = [doc for doc in raw_docs if doc.metadata.get("zodiac") == zodiac.lower()]

        # Deduplicate the documents
        seen = set()
        deduped_docs = []
        for doc in filtered_docs:
            if doc.page_content not in seen:
                deduped_docs.append(doc)
                seen.add(doc.page_content)

        # Select the top 3 documents
        top_docs = deduped_docs[:3]

        print("\n=== Retrieved Context Chunk(s) ===")
        for i, doc in enumerate(top_docs):
            print(f"\n[Chunk {i+1} - {doc.metadata.get('zodiac')}]:\n{doc.page_content}")

        # Create the context for the prompt
        context = "\n\n".join([doc.page_content for doc in top_docs])

        # Create the RAG chain
        rag_chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Invoke the RAG chain and return the response
        response = rag_chain.invoke(question)
        return response.strip()
