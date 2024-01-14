# -*- coding: utf-8 -*-
"""haha.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jaBZKIf1KhB6Qoea2urpeqirXSvc1xpG
"""


"""## Set up LLM and embedding model"""

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
from langchain_together.embeddings import TogetherEmbeddings
from langchain.llms import Together
from langchain.memory import ConversationBufferMemory
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import StorageContext, ServiceContext, load_index_from_storage, PromptHelper
from llama_index.embeddings import HuggingFaceEmbedding
import os

TOGETHER_API_KEY = "4125f82e2f0b0da68f5fcdc10766779393a5fc818b25e3872fb41b167f696c7e"

# embed_model = LangchainEmbedding(
#   HuggingFaceEmbeddings(model_name="thenlper/gte-large"),
#   embed_batch_size=32,
#   pooling="mean",
# )

llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.0,
    max_tokens=1024,
    top_k=1,
    together_api_key=TOGETHER_API_KEY
)

"""## Set up documents"""

from tqdm import tqdm
import pandas as pd
df = pd.read_json("documents.jsonl", lines=True)

from llama_index import Document, VectorStoreIndex

def get_documents():
  documents = [Document(text=t['paragraph']) for idx, t in tqdm(df.iterrows(),total=df.shape[0])]
  print(len(documents))
  return documents

from llama_index import ServiceContext, PromptHelper
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SentenceSplitter
from llama_index.prompts import PromptTemplate
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

# prompt_helper = PromptHelper(
#   context_window=3900,
#   num_output=1024,
#   chunk_overlap_ratio=0.3,
#   chunk_size_limit=None
# )

# service_context = ServiceContext.from_defaults(
#   llm=llm,
#   embed_model=embed_model,
#   prompt_helper=prompt_helper
# )

#documents = get_documents()
#index = VectorStoreIndex.from_documents(documents, service_context=service_context)

INDEX_NAME = os.environ["INDEX"]
MODEL_NAME = os.environ["MODEL_NAME"]


def get_index(model_name="thenlper/gte-large"):
  embed_model = HuggingFaceEmbedding(model_name=model_name,
  embed_batch_size=32,
  pooling="mean")

  prompt_helper = PromptHelper(
  context_window=3900,
  num_output=1024,
  chunk_overlap_ratio=0.3,
  chunk_size_limit=None
  )


  service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  prompt_helper=prompt_helper
  )

  documents = get_documents()
  index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)

  return index

#index_english = get_index()
# index_english.storage_context.persist("./index_english_new")

# index_multilingual = get_index("facebook/mcontriever-msmarco")
# index_multilingual.storage_context.persist("./index_multilingual_new")


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

# query
print(contents)
response = query_engine.query(contents)
print(response)

context_str = ""
for i, node in enumerate(response.__dict__['source_nodes']):
    context_str += "Context Text " + str(i) + ": " + node.get_text() + "\n"
prompt = """<s> [INST] <<SYS>> \nAnswer based on context only. You are allowed to use multiple contexts. \n<</SYS>>\n\n
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

 llm.invoke(prompt)

