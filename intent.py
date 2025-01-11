import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os

def ingest_csv_to_chroma(csv_path):
    messages = []
    local_embeddings = OllamaEmbeddings(model=os.environ.get("LLM_VERSION"))
    persist_directory = "db-intents"
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
        messages.append("Chrome DB загружена")
    else:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
        messages.append("Chrome DB создана")
    
    documents = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sentence = row['sentence']
            intent = row['intent']
            doc = Document(page_content=sentence, metadata={"intent": intent, "type": "intent_data"})
            documents.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)
    
    vectorstore.add_documents(documents=all_splits)
    messages.append("Данные добавлены в Chrome DB")
    
    return messages

csv_path = "data/intent.csv"

csv_messages = ingest_csv_to_chroma(csv_path)
for message in csv_messages:
    print(message)
