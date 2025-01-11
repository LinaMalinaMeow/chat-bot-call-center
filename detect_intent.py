from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(f"Sentence: {doc.page_content}\nIntent: {doc.metadata.get('intent', 'Unknown')}" for doc in docs)

def detect_intent_with_context(sentence):
    messages = []
    persist_directory = "db-intents"
    local_embeddings = OllamaEmbeddings(model="llama3.1:8b")
    
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
    
    docs = vectorstore.similarity_search(sentence)
    if not docs:
        messages.append("No relevant intent was found.")
        return {"response": "No matching intent found.", "messages": messages}
    
    print(docs);
    
    
    INTENT_TEMPLATE = """
    Ты работник коллцентра в клинике, люди пишут тебе вопросы, ты должен на них отвечать. 
    Если тебе передается вопрос, похожий на вопрос из базы знаний, отвечай на него,
    если вопрос вообще не похож на вопросы из базы знаний отвечай: Позвоните, пожалуйста, на горячую линию по номеру [+79999999999](tel:+79999999999)!
    <context>
    {context}
    </context>
    {sentence}"""
    
    intent_prompt = ChatPromptTemplate.from_template(INTENT_TEMPLATE)
    model = ChatOllama(model="llama3.1:8b")
    
    chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | intent_prompt
        | model
        | StrOutputParser()
    )
    
    response = chain.invoke({"context": docs, "sentence": sentence})
    return response
