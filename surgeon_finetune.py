import panel as pn
from langchain.chains import ConversationChain
from langchain.llms import Together
from langchain.memory import ConversationBufferMemory
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import StorageContext, ServiceContext, load_index_from_storage, PromptHelper
from llama_index.embeddings import HuggingFaceEmbedding
import os

TOGETHER_API_KEY = "df8d81058bb7142b03cf790ec619e0cde4add36dd1b731ec1eb56239f6219a46"


llm = Together(
    model="paderno@stanford.edu/Mistral-7B-Instruct-v0.2-law-2024-01-14-04-21-04",
    temperature=0.0,
    max_tokens=1024,
    top_k=1,
    together_api_key=TOGETHER_API_KEY
)

# Ensure Panel extension is loaded with required components
pn.extension()

# Define the application's title and style
title = pn.pane.Markdown("# suRAGeon FINETUNE TEST", style={'font-size': '30px', 'font-weight': 'bold', 'color': '#4a4a4a'})

# Add a logo (replace 'path_to_logo.png' with the actual path or URL to your logo)
logo = pn.pane.PNG('logo.png', width=100, height=100, align='center')

#Define the chat interface
def callback(contents: str, user: str, instance: pn.chat.ChatInterface):

   
    prompt = """<s> [INST] <<SYS>> \nAnswer the following medical question to the best of your knowledge. Do not provide any other commentary \n<</SYS>>\n\n
    Query: {query_str}
    [/INST]
    Answer:
    """.format(query_str=contents)

    print(prompt)

    print("----")

    return llm.invoke(prompt)

chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="Mistral")
chat_interface.send("Hi, I'm surgeon assistant. Please ask me your questions.", user="System", respond=False)


# Additional information panel
info_panel = pn.pane.Markdown("### Helpful Information\n Medical Advice from an assistant can be dangerous. Please beware", 
                              style={'background': '#f0f0f0', 'border': '1px solid #ddd', 'padding': '10px'})

# Footer
footer = pn.pane.Markdown("© 2024 suRAGeon. All rights reserved.", style={'text-align': 'center', 'margin-top': '20px'})

# Create the layout with title, logo, chat interface, info panel, and footer
layout = pn.Column(
    pn.Row(logo, title, align='center'),
    pn.Row(chat_interface, info_panel, align='start'),
    footer,
    background='#ffffff',
    sizing_mode='stretch_width'
)

# Make the layout servable
layout.servable()
