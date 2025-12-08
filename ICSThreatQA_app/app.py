import os
import streamlit as st
import chardet
import aiohttp
import asyncio
import pandas as pd
from io import BytesIO
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.schema import Document
import logging
import base64
from llama_index.core import KnowledgeGraphIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core import Document as D
from llama_index.core.schema import NodeWithScore
from typing import List
from llama_index.core import QueryBundle
#from transformers import pipeline

from collections import defaultdict
import json
#from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Set the page configuration first
#st.set_page_config(page_title="ICSThreat WebUI", page_icon="🔒", layout="wide", initial_sidebar_state="expanded")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HuggingFace Mistral-7B model
#llm = HuggingFaceHub(
#    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#    model_kwargs={"temperature": 0.7, "max_length": 512}
#)

# Initialize the HuggingFace model with caching
#@st.cache_resource
#def load_model(repo_id):
#    return HuggingFaceHub(
#        repo_id=repo_id,
#        model_kwargs={"temperature": 0.7, "max_length": 512}
#    )

# Step 1: User input widgets for parameters
st.sidebar.title("Model Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_length = st.sidebar.number_input("Max Length", min_value=512, max_value=2048, value=1024)

#hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#if hf_api_key is None:
#    st.error("Hugging Face API key is missing! Add it in Hugging Face secrets.")

# Step 2: Initialize the HuggingFace model with user-defined parameters
@st.cache_resource
def load_model(repo_id, temperature, max_length):    
    return HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": temperature, "max_length": max_length}
    )

# Example model initialization with user inputs
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = load_model(repo_id, temperature, max_length)

# Display the current settings to the user
#st.write(f"Model initialized with temperature: {temperature} and max_length: {max_length}")



# List of available open-source models
open_source_models = {
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-2-7B": "meta-llama/Llama-2-7b-chat-hf",
    "Zephyr-7B": "HuggingFaceH4/zephyr-7b-beta"
}


# Knowledge Graph RAG Setup
@st.cache_resource
def setup_knowledge_graph_rag(default_url):
    #os.environ["OPENAI_API_KEY"] = api_key
    # Access the API key from the environment variable
    api_key = os.environ.get('OPENAI_API_KEY')

    # Initialize the OpenAI API client
    OpenAI.api_key = api_key

    # Set up the LLM and service context for Knowledge Graph
    llm_kg = OpenAI(temperature=0, model="gpt-4o-mini-2024-07-18")
    Settings.llm = llm_kg
    Settings.chunk_size = 512
    service_context = ServiceContext.from_defaults(llm=llm_kg, chunk_size=512)

    # Load documents
    #documents = SimpleDirectoryReader(input_files=[default_url]).load_data()
    doc_list = list(default_url)
    documents = [D(text=t) for t in doc_list]

    # Set up the graph store and storage context
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # Create the Knowledge Graph Index
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=5,
        storage_context=storage_context,
        service_context=service_context
    )

    # Persist the graph
    dir_graph = r"./Data_store/graph_store"
    if not os.path.exists(dir_graph):
        os.makedirs(dir_graph)
    kg_index.storage_context.persist(persist_dir=dir_graph)

    # Load the index from storage
    storage_context = StorageContext.from_defaults(persist_dir=dir_graph, graph_store=graph_store)
    kg_index = load_index_from_storage(
        storage_context=storage_context,
        service_context=service_context,
        max_triplets_per_chunk=10,
        include_embeddings=True,
    )

    # Create the query engine
    kg_rag_query_engine = kg_index.as_query_engine(
        include_text=True,
        #retriever_mode="keyword",
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=3,
        explore_global_knowledge=True,
    )

    #Create VectorStoreIndex
    vector_index = VectorStoreIndex.from_documents(documents)

    return kg_index, vector_index, kg_rag_query_engine

# Asynchronous fetching and processing URLs
async def fetch_and_process_url(session, url):
    documents = []
    try:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch URL {url}: HTTP {response.status}")
                return documents
            
            content = await response.read()
            text = content.decode('utf-8', errors='ignore')
            documents.append(Document(page_content=text, metadata={"source": url}))
    except Exception as e:
        logger.error(f"Failed to fetch or process URL {url}: {e}")

    return documents

async def load_data_async(url):
    documents = []
    async with aiohttp.ClientSession() as session:
        documents.extend(await fetch_and_process_url(session, url))
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

# Initialize embeddings and vector store with caching
@st.cache_resource
def initialize_embeddings(_documents):
    if not _documents:
        logger.error("No documents available to embed.")
        raise ValueError("No documents available to embed.")
    
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    split_documents = text_splitter.split_documents(_documents)
    
    if not split_documents:
        logger.error("No documents available after splitting.")
        raise ValueError("No documents available after splitting.")
    
    vectorstore = FAISS.from_documents(split_documents, embedding_model)
    return vectorstore

# Default URL
#default_url = 'https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack-15.1.json'
default_url = 'https://github.com/mitre/cti/blob/master/enterprise-attack/enterprise-attack.json'
#default_url = 'https://github.com/diamond-dove/simple-json/blob/main/composer.json'

# Synchronous function to load initial data asynchronously
@st.cache_data
def load_initial_data():
    return asyncio.run(load_data_async(default_url))

# Main logic to initialize embeddings and vector store
initial_data = load_initial_data()
if not initial_data:
    raise ValueError("No documents were loaded from the provided URL.")
vectorstore_local = initialize_embeddings(initial_data)

# Define zero-shot and few-shot prompt templates for each question type
def get_zero_shot_prompt(question_type):
    templates = {
        "factual": """
        "You are a Cybersecurity expert specializing in analyzing threats and attack techniques (TTPs) in industrial control systems (ICS). Given a ICS dataset from MITRE ATT&CK, provide a precise and concise answer based solely on the critical details from the dataset. Ensure your response directly addresses the query and is tailored to the specific context of ICS attack patterns and techniques."        
        Context: {context}
        Question: {question}
        Answer:
        """,
        "contrastive": """
        "You are a Cybersecurity expert specializing in analyzing threats and attack techniques (TTPs) in industrial control systems (ICS). Given a ICS dataset from MITRE ATT&CK, provide a precise and concise answer based solely on the critical details from the dataset. Ensure your response directly addresses the query and is tailored to the specific context of ICS attack patterns and techniques."        
        Context: {context}
        Question: {question}
        Answer:
        """,
        "opinion": """
        "You are a Cybersecurity expert specializing in analyzing threats and attack techniques (TTPs) in industrial control systems (ICS). Given a ICS dataset from MITRE ATT&CK, provide a precise and concise answer based solely on the critical details from the dataset. Ensure your response directly addresses the query and is tailored to the specific context of ICS attack patterns and techniques."        
        Context: {context}
        Question: {question}
        Answer:
        """,
        "inferential": """
        "You are a Cybersecurity expert specializing in analyzing threats and attack techniques (TTPs) in industrial control systems (ICS). Given a ICS dataset from MITRE ATT&CK, provide a precise and concise answer based solely on the critical details from the dataset. Ensure your response directly addresses the query and is tailored to the specific context of ICS attack patterns and techniques."        
        Context: {context}
        Question: {question}
        Answer:
        """
    }
    return templates[question_type]

def get_few_shot_prompt(question_type):
    templates = {
        "factual": """
        You are a Cybersecurity expert specializing in analyzing threats and attack techniques (TTPs) in industrial control systems (ICS). Provide a concise and accurate answer based on the following context.
        Context: {context}
        Question: {question}
        Example:
        Context: In recent years, there has been an increase in ICS-targeted attacks leveraging unauthorized command messages.
        Question: What are the main techniques used in these attacks?
        Answer: The main techniques include manipulation of control signals, unauthorized command injection, and exploitation of vulnerable communication protocols.
        Answer:
        """,
        "contrastive": """
        You are a Cybersecurity expert specializing in analyzing threats and attack techniques (TTPs) in industrial control systems (ICS). Provide a concise and accurate answer based on the following context, highlighting the key differences or similarities.
        Context: {context}
        Question: {question}
        Example:
        Context: Both 'Triton' and 'Stuxnet' are well-known malware targeting ICS environments.
        Question: How do the attack methods of Triton differ from those of Stuxnet?
        Answer: Triton specifically targets safety instrumented systems to cause physical damage, while Stuxnet was designed to disrupt industrial processes by targeting PLCs controlling centrifuges.
        Answer:
        """,
        "opinion": """
        You are a Cybersecurity expert specializing in analyzing threats and attack techniques (TTPs) in industrial control systems (ICS). Provide a concise and well-informed opinion based on the following context.
        Context: {context}
        Question: {question}
        Example:
        Context: Many experts believe that securing remote access points in ICS environments is crucial for preventing cyber-attacks.
        Question: Do you think implementing multi-factor authentication (MFA) is sufficient to secure remote access in ICS?
        Answer: While MFA significantly enhances security, it should be complemented by network segmentation, continuous monitoring, and strict access controls to effectively secure remote access in ICS.
        Answer:
        """,
        "inferential": """
        You are a Cybersecurity expert specializing in analyzing threats and attack techniques (TTPs) in industrial control systems (ICS). Draw a logical inference based on the following context and provide a concise and accurate answer.
        Context: {context}
        Question: {question}
        Example:
        Context: An industrial facility recently experienced a sophisticated attack where adversaries manipulated control logic to cause a process disruption.
        Question: What preventive measures can be taken to safeguard against similar attacks in the future?
        Answer: Preventive measures include regular integrity checks of control logic, implementing anomaly detection systems, and ensuring that only authenticated updates can be applied to control systems.
        Answer:
        """
    }
    return templates[question_type]
    
# Function to identify question type
def identify_question_type(question):
    question = question.lower()

    # Keywords for each question type
    factual_keywords = ["what", "when", "who", "how many", "how much", "list", "name", "which", "where", "can", "is", "are", "does", "do", "was", "were"]
    contrastive_keywords = ["compare", "difference", "similar", "contrast", "versus", "vs", "differ", "similarity", "how does"]
    opinion_keywords = ["opinion", "feel", "think", "believe", "view", "suggest", "consider", "argue", "assess", "evaluate", "agree", "disagree", "recommend"]
    inferential_keywords = ["why", "how", "cause", "reason", "explain", "infer", "predict", "conclude", "result", "effect", "impact", "what if", "implication", "consequence"]

    # Checking for complex phrases and structures
    if any(phrase in question for phrase in contrastive_keywords):
        return "contrastive"
    elif any(phrase in question for phrase in opinion_keywords):
        return "opinion"
    elif any(phrase in question for phrase in inferential_keywords):
        return "inferential"
    elif any(phrase in question for phrase in factual_keywords):
        return "factual"

    # Handle cases where multiple types could apply, consider the structure of the question
    if "or" in question or "vs" in question:
        return "contrastive"
    if "should" in question:
        return "opinion"

    # Default to factual if no clear indicators are found
    return "factual"


# Keyword extraction function
def extract_keywords(query, model_name="Zephyr-7B"):
    model = load_model(open_source_models[model_name], temperature, max_length)
    prompt = f"Identify and extract the most critical keywords from the following query that are essential for retrieving the most relevant and accurate information. Focus on key terms, entities, and specific phrases that will lead to the best possible answer: '{query}'"
    response = model.generate(prompts=[prompt])
    keywords = response.generations[0][0].text.strip().split(", ")
    return keywords

# Retrieve and generate keyword-based answers
def retrieve_and_generate_keyword_based_answers(query, vectorstore, model_name="Zephyr-7B", question_type="factual"):
    keywords = extract_keywords(query, model_name)
    keyword_query = " ".join(keywords)
    
    prompt_template_zero_shot = get_zero_shot_prompt(question_type)
    prompt_template_few_shot = get_few_shot_prompt(question_type)
    
    # Zero-shot answer
    qa_chain_zero_shot = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template_zero_shot, input_variables=["context", "question"])}
    )
    response_zero_shot = qa_chain_zero_shot.invoke({"query": keyword_query})
    zero_shot_result = response_zero_shot['result']
    zero_shot_sources = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in response_zero_shot['source_documents']]
    zero_shot_contexts = [doc.page_content for doc in zero_shot_sources]
    
    # Few-shot answer
    qa_chain_few_shot = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template_few_shot, input_variables=["context", "question"])}
    )
    response_few_shot = qa_chain_few_shot.invoke({"query": keyword_query})
    few_shot_result = response_few_shot['result']
    few_shot_sources = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in response_few_shot['source_documents']]
    few_shot_contexts = [doc.page_content for doc in few_shot_sources]
    
    return zero_shot_result, zero_shot_sources, zero_shot_contexts, few_shot_result, few_shot_sources, few_shot_contexts


#def trim_and_clean_response(response):
    # Trim the response to keep it concise and follow the specified format
 #   if "Answer:" in response:
        # Extract the part starting after "Answer:"
  #      trimmed_response = response.split("Answer:")[1].strip()
   # else:
    #    trimmed_response = response.strip()
    #return trimmed_response

def trim_and_clean_response(response):
    # Trim the response to keep it concise and follow the specified format
    if "Answer:" in response:
        # Split the response by "Answer:" to handle multiple examples
        all_answers = response.split("Answer:")
        
        # Assuming the last "Answer:" part is the response we want to extract
        trimmed_response = all_answers[-1].strip()
    else:
        trimmed_response = response.strip()
    
    return trimmed_response
    
# Retrieve and generate zero-shot answers
def retrieve_and_generate_zero_shot_answers(query, vectorstore, question_type):
    prompt_template = get_zero_shot_prompt(question_type)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    )
    response = qa_chain.invoke({"query": query})
    if not response['result'].strip():  # Check if the result is empty or only whitespace
        return "Sorry, I don't know.", [], []
    sources = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in response['source_documents']]
    contexts = [doc.page_content for doc in sources]
    return response['result'], sources, contexts

# Retrieve and generate few-shot answers
def retrieve_and_generate_few_shot_answers(query, vectorstore, question_type):
    prompt_template = get_few_shot_prompt(question_type)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    )
    response = qa_chain.invoke({"query": query})
    if not response['result'].strip():  # Check if the result is empty or only whitespace
        return "Sorry, I don't know.", [], []
    sources = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in response['source_documents']]
    contexts = [doc.page_content for doc in sources]
    return response['result'], sources, contexts
  
# Retrieve and generate Knowledge Graph-based answers
def retrieve_and_generate_kg_rag_answers(query):
    response_graph_rag = kg_rag_query_engine.query(query)
    res_rag=f"<b>{response_graph_rag}</b>"
    context_rag=response_graph_rag
    return str(res_rag), context_rag  # Return as text since no sources or contexts are provided in this example


# Function to generate answer using selected LLM
def generate_answer_with_llm(query, selected_model, question_type=None):
    model = load_model(selected_model, temperature, max_length)
    prompt_template_zero_shot = get_zero_shot_prompt(question_type)
    prompt_template_few_shot = get_few_shot_prompt(question_type)
    context = "This is a placeholder context. Replace with actual context if available."  # Replace with actual context if available
    formatted_query_zero_shot = prompt_template_zero_shot.format(context=context, question=query)
    formatted_query_few_shot = prompt_template_few_shot.format(context=context, question=query)
    
    response_zero_shot = model.generate(prompts=[formatted_query_zero_shot])
    response_few_shot = model.generate(prompts=[formatted_query_few_shot])
    
    return response_zero_shot.generations[0][0].text, response_few_shot.generations[0][0].text

# New function to retrieve and generate answers using the combined method
def retrieve_and_generate_combined_answers(query, vectorstore, question_type="factual"):
    
    model_name="Zephyr-7B"
    # Get the context from keyword-based retrieval
    keywords = extract_keywords(query, model_name)
    keyword_query = " ".join(keywords)
    
    prompt_template_zero_shot = get_zero_shot_prompt(question_type)
    prompt_template_few_shot = get_few_shot_prompt(question_type)
    
    # Perform keyword-based retrieval
    qa_chain_keyword = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template_zero_shot, input_variables=["context", "question"])}
    )
    response_keyword = qa_chain_keyword.invoke({"query": keyword_query})
    keyword_contexts = [doc.page_content for doc in response_keyword['source_documents']]
    
    # Perform embedding-based retrieval
    qa_chain_embedding = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template_zero_shot, input_variables=["context", "question"])}
    )
    response_embedding = qa_chain_embedding.invoke({"query": query})
    embedding_contexts = [doc.page_content for doc in response_embedding['source_documents']]
    
    # Combine contexts from both methods
    zero_shot_combined_contexts = keyword_contexts + embedding_contexts
    zero_shot_combined_context = " ".join(zero_shot_combined_contexts)

    # Generate zero-shot answer
    combined_zero_shot_query = prompt_template_zero_shot.format(context=zero_shot_combined_context, question=query)
    combined_zero_shot_answer = llm.generate(prompts=[combined_zero_shot_query]).generations[0][0].text


    # Perform keyword-based retrieval
    qa_chain_keyword = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template_few_shot, input_variables=["context", "question"])}
    )
    response_keyword = qa_chain_keyword.invoke({"query": keyword_query})
    keyword_contexts = [doc.page_content for doc in response_keyword['source_documents']]
    
    # Perform embedding-based retrieval
    qa_chain_embedding = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template_few_shot, input_variables=["context", "question"])}
    )
    response_embedding = qa_chain_embedding.invoke({"query": query})
    embedding_contexts = [doc.page_content for doc in response_embedding['source_documents']]

    # Combine contexts from both methods
    few_shot_combined_contexts = keyword_contexts + embedding_contexts
    few_shot_combined_context = " ".join(zero_shot_combined_contexts)

    # Generate zero-shot answer
    combined_few_shot_query = prompt_template_few_shot.format(context=few_shot_combined_context, question=query)
    combined_few_shot_answer = llm.generate(prompts=[combined_few_shot_query]).generations[0][0].text
    
    
    return trim_and_clean_response(combined_zero_shot_answer), zero_shot_combined_contexts, trim_and_clean_response(combined_few_shot_answer), few_shot_combined_contexts

# Load a pre-trained model for sentence embeddings
embedding_model_1 = SentenceTransformer('all-MiniLM-L6-v2')

# Function to calculate similarity
def calculate_similarity(query, response):
    response = str(response) if not isinstance(response, str) else response
    query_embedding = embedding_model_1.encode([query])
    response_embedding = embedding_model_1.encode([response])
    similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
    return similarity


def weighted_fusion(query, vector_response, kg_response):
    # Calculate similarity scores
    vector_score = calculate_similarity(query, vector_response)
    kg_score = calculate_similarity(query, kg_response)

    #repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    #model = load_model(repo_id, temperature, max_length)
    #generator = pipeline("text2text-generation", model="google/flan-t5-large")

    # Normalize scores to get weights
    total_score = vector_score + kg_score
    vector_weight = vector_score / total_score
    kg_weight = kg_score / total_score

    # Simple LLM prompt for merging
    prompt = f"""
    Given the following responses, merge them into a concise, coherent, and informative final response based on Vector Response and Knowledge Graph Response Weights.
    
    Query: "{query}"
    Vector Response (weight {vector_weight:.2f}): "{vector_response}"
    Knowledge Graph Response (weight {kg_weight:.2f}): "{kg_response}"
    
    Consider the relevance weights to emphasize key information. 

    Final Merged Response:
    """

     # Generate the merged response
    #merged_response = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
    response = llm.generate(prompts=[prompt])
    merged_response = response.generations[0][0].text.strip().split(", ")

    # Extract the generated merged response after the prompt
    #merged_response = merged_response.split("Merged Response:")[1].strip()

    return merged_response


# Function to improve response using Vector Store and Knowledge Graph
def improve_response_with_feedback(query, response,vector_index, kg_index):
    #with st.spinner("🔄 Improving the response based on your feedback..."):
     # Set up individual retrievers
        #vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)
        #keyword_retriever = KeywordTableSimpleRetriever(index=kg_index)

        # Custom Retriever (AND/OR mode)
        #custom_retriever = CustomRetriever(vector_retriever, keyword_retriever, mode="OR")

        # Response Synthesizer
        #response_synthesizer = get_response_synthesizer()

        # Query Engine using the custom retriever
        #query_engine = RetrieverQueryEngine(
         #   retriever=custom_retriever,
          #  response_synthesizer=response_synthesizer,
        #)

        # Execute the query
        #improved_response = query_engine.query(query)

        # Display the improved response
        #st.success("✨ Improved Response Ready!")
        #st.markdown(f"<div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>{improved_response}</div>", unsafe_allow_html=True)

        #return improved_response
    with st.spinner("🔄 Improving the response based on your feedback..."):
        # Ensure the response is in the correct format if necessary
        # For example, if 'response' needs to be a Document or Node, convert it here
    
        # Vector Store Query Engine
        vector_query_engine = vector_index.as_query_engine()
            
        # Knowledge Graph Query Engine
        kg_query_engine = kg_index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=3,
                explore_global_knowledge=True,
        )
    
        # Perform queries
        #vector_response = vector_query_engine.query(query)
        #kg_response = kg_query_engine.query(query)

        # Get the actual text content from the response objects
        vector_response_obj = vector_query_engine.query(query)
        kg_response_obj = kg_query_engine.query(query)

        # Access the text (assuming the text is in the `.response` attribute)
        vector_response = vector_response_obj.response if hasattr(vector_response_obj, 'response') else str(vector_response_obj)
        kg_response = kg_response_obj.response if hasattr(kg_response_obj, 'response') else str(kg_response_obj)

    
        # Combine responses
        #improved_response = f"{vector_response.response}\n{kg_response.response}"
        # Contextual merging using weighted fusion
        improved_response = weighted_fusion(query, vector_response, kg_response)
        
            
        # Display the improved response
        st.write("**Improved Response after Feedback:**")
        st.success("✨ Improved Response Ready!")
        st.markdown(
        f"<div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>{improved_response}</div>",
                unsafe_allow_html=True
        )
            
        return improved_response

# Online Learning-based KG-RAG function
#def online_learning_kg_rag(query, kg_rag_query_engine):
def online_learning_kg_rag(query,vector_index, kg_index):
    # Phase 1: Retrieve initial responses
    raw_response = kg_rag_query_engine.query(query)
    responses = [raw_response.response]  # List to handle multiple responses if applicable

    # Phase 2: Rank responses based on feedback
    #ranked_responses = rank_responses_based_on_feedback(responses, query)


    # Initialize session state for feedback tracking
    if 'feedback_given' not in st.session_state:
        st.session_state['feedback_given'] = False

    # Display the KG-RAG Response
    st.write("**KG-RAG Response:**")
    st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{responses[0]}</div>", unsafe_allow_html=True)

    # Callback functions to handle feedback
    def positive_feedback():
        st.session_state['feedback_given'] = 'positive'
        st.success("✅ Thank you for your positive feedback!")
    
    def negative_feedback(query, response, vector_index, kg_index):
        st.session_state['feedback_given'] = 'negative'
        st.warning("🔄 Improving the response based on your feedback...")
        improved_response = improve_response_with_feedback(query, responses[0],vector_index, kg_index)
        #st.markdown(f"<div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>{improved_response}</div>", unsafe_allow_html=True)
        return improved_response

    def handle_negative_feedback():
        nonlocal improved_response  # Ensure we update the outer variable
        improved_response = negative_feedback(query, responses, vector_index, kg_index)
    
    # Display feedback buttons
    col1, col2 = st.columns(2)
    improved_response = None  # To store improved response
    
    with col1:
        st.button("👍", key="positive_feedback", on_click=positive_feedback)
    with col2:
        #st.button("👎", key="negative_feedback", on_click=negative_feedback)
        st.button("👎", key="negative_feedback", on_click=handle_negative_feedback)

    # Display feedback status
    #if st.session_state['feedback_given']:
     #   st.write(f"Feedback received: {st.session_state['feedback_given']}")
    if st.session_state['feedback_given'] == 'negative':
        return improved_response if improved_response else responses
    else:
        return responses



# Handle user query or file paths
def handle_user_query(query, selected_model=None, use_keywords=False, use_combined=False, use_kg_rag=False, use_online_learning=False):
    question_type = identify_question_type(query)

    if use_kg_rag:
        zero_shot_answer,zero_shot_context = retrieve_and_generate_kg_rag_answers(query)
        zero_shot_answer = trim_and_clean_response(zero_shot_answer)
        #few_shot_answer = trim_and_clean_response(few_shot_answer)
        #zero_shot_sources = few_shot_sources = []
        return zero_shot_answer,zero_shot_context
  
    elif use_combined:
        zero_shot_answer, zero_shot_contexts, few_shot_answer, few_shot_contexts = retrieve_and_generate_combined_answers(query, vectorstore_local, question_type=question_type)
        zero_shot_answer = trim_and_clean_response(zero_shot_answer)
        few_shot_answer = trim_and_clean_response(few_shot_answer)
        return zero_shot_answer, zero_shot_contexts, few_shot_answer, few_shot_contexts
        #zero_shot_sources = []
        #few_shot_sources = []
    
    elif selected_model:
        zero_shot_answer, few_shot_answer = generate_answer_with_llm(query, selected_model, question_type)
        zero_shot_answer = trim_and_clean_response(zero_shot_answer)
        few_shot_answer = trim_and_clean_response(few_shot_answer)
        return zero_shot_answer,few_shot_answer
        #zero_shot_sources = []
        #few_shot_sources = []
        #zero_shot_contexts = []
        #few_shot_contexts = []
    
    elif use_keywords:
        zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts = retrieve_and_generate_keyword_based_answers(query, vectorstore_local, question_type=question_type)
        zero_shot_answer = trim_and_clean_response(zero_shot_answer)
        few_shot_answer = trim_and_clean_response(few_shot_answer)

    elif use_online_learning:
        #online_learning_kg_rag(query, kg_rag_query_engine)
        zero_shot_answer = online_learning_kg_rag(query,vector_index, kg_index)
        zero_shot_answer = trim_and_clean_response(zero_shot_answer)
        return zero_shot_answer
    else:
        zero_shot_answer, zero_shot_sources, zero_shot_contexts = retrieve_and_generate_zero_shot_answers(query, vectorstore_local, question_type)
        zero_shot_answer = trim_and_clean_response(zero_shot_answer)
        few_shot_answer, few_shot_sources, few_shot_contexts = retrieve_and_generate_few_shot_answers(query, vectorstore_local, question_type)
        few_shot_answer = trim_and_clean_response(few_shot_answer)
    
    return zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts



def process_csv_file(file_path, vectorstore, selected_model=None, use_keywords=False, use_combined=False, use_kg_rag=False, use_online_learning=False):
    try:
        # Detect file encoding
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        
        df = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        st.error(f"An error occurred while reading the CSV file: {e}")
        return

    question_types = []
    zero_shot_answers = []
    few_shot_answers = []
    zero_shot_sources_list = []
    few_shot_sources_list = []
    zero_shot_contexts_list = []
    few_shot_contexts_list = []

    for idx, row in df.iterrows():
        question = row['questions']
        question_type = identify_question_type(question)
        question_types.append(question_type)

        if use_kg_rag:
            zero_shot_answer, zero_shot_context = handle_user_query(question, use_kg_rag=True)
            zero_shot_answers.append(zero_shot_answer)
            # Ensure other lists also maintain the same length
            few_shot_answers.append("")  # Append empty strings or None
            zero_shot_sources_list.append("")
            few_shot_sources_list.append("")
            zero_shot_contexts_list.append(zero_shot_context)
            few_shot_contexts_list.append("")
        elif use_online_learning:
            zero_shot_answer = handle_user_query(input_text, use_online_learning=True)
            
        else:        
            # Use the handle_user_query function to process each question
            zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts = handle_user_query(
                question, 
                selected_model=selected_model, 
                use_keywords=use_keywords, 
                use_combined=use_combined,
                use_kg_rag=use_kg_rag
            )
            
            zero_shot_answers.append(zero_shot_answer)
            few_shot_answers.append(few_shot_answer)
            zero_shot_sources_list.append("; ".join([f"source: {doc.metadata.get('source')}" for doc in zero_shot_sources]) if zero_shot_sources else "")
            few_shot_sources_list.append("; ".join([f"source: {doc.metadata.get('source')}" for doc in few_shot_sources]) if few_shot_sources else "")
            zero_shot_contexts_list.append(" ".join(zero_shot_contexts) if zero_shot_contexts else "")
            few_shot_contexts_list.append(" ".join(few_shot_contexts) if few_shot_contexts else "")
    
    # Ensure DataFrame creation matches all lists
    if use_kg_rag:
        result_df = pd.DataFrame({
            'questions': df['questions'],
            'question_type': question_types,
            'zero_shot_answers': zero_shot_answers,
            'zero_shot_contexts': zero_shot_contexts_list
        })
    elif use_online_learning:
        result_df = pd.DataFrame({
            'questions': df['questions'],
            'question_type': question_types,
            'zero_shot_answers': zero_shot_answers
        })        
    else:
        result_df = pd.DataFrame({
            'questions': df['questions'],
            'question_type': question_types,
            'zero_shot_answers': zero_shot_answers,
            'zero_shot_sources': zero_shot_sources_list,
            'zero_shot_contexts': zero_shot_contexts_list,
            'few_shot_answers': few_shot_answers,
            'few_shot_sources': few_shot_sources_list,
            'few_shot_contexts': few_shot_contexts_list
        })

    try:
        result_df.to_csv(file_path, index=False, encoding='utf-8')
        st.success("CSV file processed and updated successfully.")
        
        csv = result_df.to_csv(index=False, encoding='utf-8')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="processed_results.csv">Download processed CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while saving the CSV file: {e}")


def format_source_document(doc):
    source_html = f"""
    <div style='padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #e8f4ff;'>
        <p><strong>Source:</strong> <a href="{doc.metadata.get('source')}" target="_blank">{doc.metadata.get('source')}</a></p>
    </div>
    """
    return source_html

def display_answer(zero_shot_answer, zero_shot_sources=None, zero_shot_contexts=None, few_shot_answer=None, few_shot_sources=None, few_shot_contexts=None):
    # Display Zero-Shot Answer
    st.write("**Zero-Shot Answer:**")
    st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{zero_shot_answer}</div>", unsafe_allow_html=True)
    
    # Display Zero-Shot Context if available
    if zero_shot_contexts:
        st.write("**Zero-Shot Context:**")
        st.markdown(f"<div style='background-color: #e8f4ff; padding: 10px; border-radius: 5px;'>{' '.join(zero_shot_contexts)}</div>", unsafe_allow_html=True)
    
    # Display Zero-Shot Source Documents if available
    #if zero_shot_sources:
    #    st.write("**Zero-Shot Source Documents:**")
    #    for doc in zero_shot_sources:
    #        st.markdown(format_source_document(doc), unsafe_allow_html=True)
    
    # Display Few-Shot Answer if available
    if few_shot_answer:
        st.write("**Few-Shot Answer:**")
        st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{few_shot_answer}</div>", unsafe_allow_html=True)
    
    # Display Few-Shot Context if available
    if few_shot_contexts:
        st.write("**Few-Shot Context:**")
        st.markdown(f"<div style='background-color: #e8f4ff; padding: 10px; border-radius: 5px;'>{' '.join(few_shot_contexts)}</div>", unsafe_allow_html=True)
    
    # Display Few-Shot Source Documents if available
    #if few_shot_sources:
    #    st.write("**Few-Shot Source Documents:**")
    #    for doc in few_shot_sources:
    #        st.markdown(format_source_document(doc), unsafe_allow_html=True)

def process_single_query(approach, input_text, selected_model=None):
    try:
        if approach == "Retrieval Augmentation Generation (RAG)":
            zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts = handle_user_query(input_text)
            display_answer(zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts)
        elif approach == "Large Language Model (LLM)":
            #zero_shot_answer, few_shot_answer = generate_answer_with_llm(input_text, selected_model,question_type=question_type)
            zero_shot_answer, few_shot_answer = handle_user_query(input_text,selected_model=selected_model)
            #zero_shot_answer, _, _, few_shot_answer, _, _ = handle_user_query(input_text, selected_model=selected_model)
            #zero_shot_sources = few_shot_sources = zero_shot_contexts = few_shot_contexts = []
            #display_answer(zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts)
            display_answer(zero_shot_answer, few_shot_answer)
        elif approach == "Keyword-Based Retrieval":
            zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts = handle_user_query(input_text, use_keywords=True)
            display_answer(zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts)
        elif approach == "Combined Retrieval Method":
            #zero_shot_answer, zero_shot_contexts, few_shot_answer, few_shot_contexts = retrieve_and_generate_combined_answers(input_text, vectorstore_local)
            zero_shot_answer, zero_shot_contexts, few_shot_answer, few_shot_contexts = handle_user_query(input_text, use_combined=True)
            #zero_shot_answer, _, zero_shot_contexts, few_shot_answer, _, few_shot_contexts = handle_user_query(input_text, use_combined=True)
            #zero_shot_sources = few_shot_sources = []
            #display_answer(zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts)
            display_answer(zero_shot_answer,zero_shot_contexts, few_shot_answer, few_shot_contexts)
        elif approach == "Knowledge Graph RAG":
            #zero_shot_answer, zero_shot_contexts = retrieve_and_generate_kg_rag_answers(input_text)
            zero_shot_answer, zero_shot_contexts = handle_user_query(input_text, use_kg_rag=True)
            display_answer(zero_shot_answer, zero_shot_contexts)
        elif approach == "Online Learning-based KG RAG":
            handle_user_query(input_text, use_online_learning=True)
        #display_answer(zero_shot_answer, zero_shot_sources, zero_shot_contexts, few_shot_answer, few_shot_sources, few_shot_contexts)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def process_bulk_query(approach, uploaded_file, selected_model=None):
    if uploaded_file:
        try:
            file_path = uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            #save_path = st.text_input("Enter the path to save the processed CSV results file:")
            if approach == "Retrieval Augmentation Generation (RAG)":
                process_csv_file(file_path, vectorstore_local)
            elif approach == "Large Language Model (LLM)":
                process_csv_file(file_path, vectorstore_local, selected_model=selected_model)
            elif approach == "Keyword-Based Retrieval":
                process_csv_file(file_path, vectorstore_local, use_keywords=True)
            elif approach == "Combined Retrieval Method":
                process_csv_file(file_path, vectorstore_local, use_combined=True)
            elif approach == "Knowledge Graph RAG":
                process_csv_file(file_path, vectorstore_local, use_kg_rag=True)
            elif approach == "Online Learning-based KG RAG":
                process_csv_file(file_path, vectorstore_local, use_online_learning=True)
        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {e}")

def main():
    st.title("ICSThreatQA Web UI")
    st.write("Your go-to tool for analyzing and understanding attackers TTPs and threats in ICS.")

    st.markdown('<div class="instructions"><p><strong>Instructions:</strong></p>'
            '<ul>'
            '<li>Select a model from the dropdown list and choose query type.</li>'
            '<li>Enter your query if using a single query or upload a CSV file for bulk queries.</li>'
            '<li>Adjust the temperature (0 for deterministic, 1 for creative).</li>'
            '<li>Click "Get Answer" for a single query or "Process CSV" for bulk queries.</li>'
            '</ul></div>', unsafe_allow_html=True)


    st.sidebar.write("""
    - Examples:
    - What are the latest tactics used in ICS attacks?
    - How can we defend against lateral movement in ICS?
    - Compare the attack techniques used in Triton and Stuxnet.
    - What are the potential impacts of a phishing attack on ICS?
    """)

    # Step 1: Choose between RAG, LLM, Keyword-Based Retrieval, and Combined Retrieval
    approach = st.selectbox("Select Model:", ["Retrieval Augmentation Generation (RAG)", "Large Language Model (LLM)", "Keyword-Based Retrieval", "Combined Retrieval Method", "Knowledge Graph RAG", "Online Learning-based KG RAG"])

    # Step 2: Choose single query or bulk queries
    query_type = st.selectbox("Select single/bulk query option:", ["Single query", "Bulk queries (CSV file)"])

    #Step 3: Choose question type from the list
    #question_type = st.selectbox("Select question type:", ["factual", "contrastive", "opinion", "inferential"])

    if query_type == "Single query":
        input_text = st.text_area("Enter your question:")
        if approach == "Large Language Model (LLM)":
            selected_model = st.selectbox("Select Open-Source LLM:", ["None"] + list(open_source_models.keys()))
            #selected_model_repo = open_source_models[selected_model] if selected_model != "None" else None
            selected_model_repo = open_source_models[selected_model] if selected_model != "None" else None

            if st.button("Get Answer"):
                if selected_model_repo:
                    process_single_query(approach, input_text, selected_model_repo)
                else:
                    st.error("No valid model selected. Please choose a valid LLM.")
        else:
            if st.button("Get Answer"):
                process_single_query(approach, input_text)
        #if st.button("Get Answer"):
         #   if approach == "Large Language Model (LLM)":
          #      selected_model = st.selectbox("Select Open-Source LLM:", ["None"] + list(open_source_models.keys()))
           #     selected_model_repo = open_source_models[selected_model] if selected_model != "None" else None
            #    process_single_query(approach, input_text, selected_model_repo)
            #else:
             #   process_single_query(approach, input_text)
    else:  # Bulk queries
        uploaded_file = st.file_uploader("Choose a CSV file with questions", type="csv")
        if approach == "Large Language Model (LLM)":
            selected_model = st.selectbox("Select Open-Source LLM:", ["None"] + list(open_source_models.keys()))
            #selected_model_repo = open_source_models[selected_model] if selected_model != "None" else None
            selected_model_repo = open_source_models[selected_model] if selected_model != "None" else None
            if st.button("Process CSV"):
                if selected_model_repo:
                    process_bulk_query(approach, uploaded_file, selected_model=selected_model_repo)
                else:
                    st.error("No valid model selected. Please choose a valid LLM.")
        
        else:
            if st.button("Process CSV"):
                process_bulk_query(approach, uploaded_file)
        #if st.button("Process CSV"):
         #   if approach == "Large Language Model (LLM)":
          #      selected_model = st.selectbox("Select Open-Source LLM:", ["None"] + list(open_source_models.keys()))
           #     selected_model_repo = open_source_models[selected_model] if selected_model != "None" else None
            #    process_bulk_query(approach, uploaded_file, selected_model=selected_model_repo)
            #else:
             #   process_bulk_query(approach, uploaded_file)

if __name__ == '__main__':
    kg_index, vector_index, kg_rag_query_engine = setup_knowledge_graph_rag(default_url)
    main() 