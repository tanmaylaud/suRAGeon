import panel as pn
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Ensure Panel extension is loaded with required components
pn.extension()

# Define the application's title and style
title = pn.pane.Markdown("# suRAGeon", style={'font-size': '30px', 'font-weight': 'bold', 'color': '#4a4a4a'})

# Add a logo (replace 'path_to_logo.png' with the actual path or URL to your logo)
logo = pn.pane.PNG('logo.png', width=100, height=100, align='center')

# Define the chat interface
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    # Validate user input here if necessary
    await chain.apredict(input=contents)

chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="ChatGPT")
chat_interface.send("Hi, I'm surgeon assistant. Please ask me your questions.", user="System", respond=False)

# Set up the callback handler, language model, and memory
callback_handler = pn.chat.langchain.PanelCallbackHandler(chat_interface)
llm = ChatOpenAI(streaming=True, callbacks=[callback_handler])
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

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
