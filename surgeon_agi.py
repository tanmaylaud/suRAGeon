import panel as pn
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Ensure Panel extension is loaded with required components
pn.extension()

# Define the application's title and style
title = pn.pane.Markdown("# suRAGeon", style={'font-size': '30px', 'font-weight': 'bold'})

# Add a logo (replace 'path_to_logo.png' with the actual path or URL to your logo)
logo = pn.pane.PNG('logo.png', width=100, height=100, align='center')

# Define the chat interface
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    await chain.apredict(input=contents)

chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="ChatGPT")
chat_interface.send("Hi, I'm surgeon assistant. Please ask me your questions.", user="System", respond=False)

# Set up the callback handler, language model, and memory
callback_handler = pn.chat.langchain.PanelCallbackHandler(chat_interface)
llm = ChatOpenAI(streaming=True, callbacks=[callback_handler])
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

# Create the layout with title, logo, and chat interface
layout = pn.Column(pn.Row(logo, title, align='center'), chat_interface)

# Make the layout servable
layout.servable()
