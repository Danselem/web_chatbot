from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate, HuggingFaceHub

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Import gradio for UI
import gradio as gr


# Setting Environmental variables and API Key
os.environ["TOKENIZERS_PARALLELISM"] = "true"

with open('hf_api.txt') as f:
    hf_key = f.readlines()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key[0]
    
# Defining data directory to persist
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Loading the model
def load_llm():
    model_id = "declare-lab/flan-alpaca-large"
    llm = HuggingFaceHub(
        repo_id=model_id,
        model_kwargs={"temperature":0.2, "max_length":1024}
        )
    return llm

def load_memory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory

def embed():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'mps'})
    return embeddings

def load_vector(embedding):
    db = FAISS.load_local(DB_FAISS_PATH, embedding)
    return db

def generate(prompt): 
    # The prompt will get passed to the LLM Chain!
    result = chat({"question": prompt})
    
    return result['answer']

def my_chatbot(input, history):
    history = history or []
    my_history = list(sum(history, ()))
    my_history.append(input)
    my_input = ' '.join(my_history)
    output = generate(my_input)
    history.append((input, output))
    return history, history 



if __name__ == '__main__':
    
    qa_prompt = set_custom_prompt()
    memory = load_memory()
    embeddings = embed()
    vectorstore = load_vector(embeddings)
    llm = load_llm()
    retriever=vectorstore.as_retriever()
    chat = ConversationalRetrievalChain.from_llm(llm,retriever=retriever,memory=memory)
    
    # Define a string variable to hold the title of the app
    title = '🦜🔗 Web 3 Chatbot'
    # Define another string variable to hold the description of the app
    description = 'This application demonstrates the use of the open-source LLM for Web 3 Chatbot.'
    
    with gr.Blocks(theme=gr.themes.Glass()) as demo:
        gr.Markdown("""<h1><center>🦜🔗 Web 3 Chatbot</center></h1>""")
        gr.Markdown("""<h3><center>🦜🔗 This application demonstrates the use of open-source LLM for Web 3 Chatbot</center></h3>""")
        chatbot = gr.Chatbot()
        state = gr.State()
        txt = gr.Textbox(show_label=False, placeholder="Ask me a question and press enter.")
        clear = gr.ClearButton([txt, chatbot])
        txt.submit(my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
    
    demo.launch(server_port=8080, share = True)
