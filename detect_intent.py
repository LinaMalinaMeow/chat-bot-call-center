from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(f"Sentence: {doc.page_content}\nIntent: {doc.metadata.get('intent', 'Unknown')}" for doc in docs)


def detect_intent_with_context(sentence):
    persist_directory = "db-intents"
    local_embeddings = OllamaEmbeddings(model="llama3.1:8b")
    
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)
    
    docs = vectorstore.similarity_search_with_score(sentence)
    filtredDocs = []
    
    for doc,score in docs:
        if score <= 0.25:
            filtredDocs.append(doc)

    if not filtredDocs:
        return "Позвоните, пожалуйста, на горячую линию по номеру +79999999999!"
    

    INTENT_TEMPLATE = """
    Ты — человек, консультант клиники, который помогает находить информацию в базе данных и отвечать на вопросы.
    Вот данные, которые могут содержать полезные сведения по заданному вопросу:
    <context>
    {context}
    <context>
    Вопрос: {sentence}
    Если вопрос не имеет точную формулировку как в базе знаний, но означает ее, отвечай на него.
    Если считаешь, что вопроса нет в базе знаний, отвечай: "Позвоните, пожалуйста, на горячую линию по номеру +79999999999!"
    Ответ должен быть кратким и состоять из информации, соответствующей содержащейся информации.
    И не пиши, что это ответ на вопрос, пиши просто ответ на вопрос
    """
    
    intent_prompt = ChatPromptTemplate.from_template(INTENT_TEMPLATE)
    model = ChatOllama(model="llama3.1:8b")
    
    chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | intent_prompt
        | model
        | StrOutputParser()
    )
    
    response = chain.invoke({"context": filtredDocs, "sentence": sentence})
    return response
