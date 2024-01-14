import panel as pn
from langchain.chains import ConversationChain
from langchain.llms import Together
from langchain.memory import ConversationBufferMemory
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import StorageContext, ServiceContext, load_index_from_storage, PromptHelper
from llama_index.embeddings import HuggingFaceEmbedding
import os

TOGETHER_API_KEY = "4125f82e2f0b0da68f5fcdc10766779393a5fc818b25e3872fb41b167f696c7e"


llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.0,
    max_tokens=1024,
    top_k=1,
    together_api_key=TOGETHER_API_KEY
)


INDEX_NAME = os.environ["INDEX"]
MODEL_NAME = os.environ["MODEL_NAME"]
# Ensure Panel extension is loaded with required components
pn.extension()

# Define the application's title and style
title = pn.pane.Markdown("# suRAGeon", style={'font-size': '30px', 'font-weight': 'bold', 'color': '#4a4a4a'})

# Add a logo (replace 'path_to_logo.png' with the actual path or URL to your logo)
logo = pn.pane.PNG('logo.png', width=100, height=100, align='center')

sc = StorageContext.from_defaults(persist_dir=INDEX_NAME)
embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME, pooling="mean")
prompt_helper = PromptHelper(
context_window=4096,
num_output=1024,
chunk_overlap_ratio=0.3,
chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model, prompt_helper=prompt_helper)
index = load_index_from_storage(storage_context=sc, service_context=service_context)

 # Validate user input here if necessary
retriever = VectorIndexRetriever(
index=index,
similarity_top_k=5)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
)
# Define the chat interface
def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
   
    # query
    print(contents)
    response = query_engine.query(contents)
    print(response)

    context_str = ""
    for i, node in enumerate(response.__dict__['source_nodes']):
        context_str += "Context Text " + str(i) + ": " + node.get_text() + "\n"
    prompt = """<s> [INST] <<SYS>> \nAnswer based on context only. Do not provide any other commentary \n<</SYS>>\n\n
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and no prior knowledge, answer the query.
    Query: {query_str}
    [/INST]
    Answer:
    """.format(context_str=context_str, query_str=contents)

    print(prompt)

    print("----")

    prompts_dict = query_engine.get_prompts()
    print(prompts_dict)

    response =  llm.invoke(prompt)

    contexts = "#### Evidences: \n" +context_str
    chat_interface.send(contexts, user="System", respond=False)
    return response

chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="Mistral")
chat_interface.send("Hi, I'm surgeon assistant. Please ask me your questions.", user="System", respond=False)


# Additional information panel
info_panel = pn.pane.Markdown("### Helpful Information\nHere are some tips to help you get started...", 
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
