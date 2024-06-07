import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback

from modules.llm.llm_settings import LLMSettings

#fix Error: module 'langchain' has no attribute 'verbose'
import langchain
langchain.verbose = False

class SinglePDFChatbot:

    def __init__(self, model_name, llm_settings: LLMSettings):
        self.model_name = model_name
        self.llm_settings = LLMSettings()

    qa_template_en = """
    
        You are a helpful AI assistant named Alago. The user provides you with a file whose content is represented by the following context parts. Use these to answer the question at the end.
        The files deal with topics from the construction industry. You can assume that the user is a civil engineer or architect.
        If you don't know the answer, just say that you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you only answer questions related to the context.
        Provide as detailed an answer as possible.

        <context>
        {context}
        </context>
        
        <question>
        {question}
        </question>
    
    """
    
    qa_template = """
        Du bist ein hilfreicher KI-Assistent namens Alago. Der Benutzer gibt Ihnen eine Datei, deren Inhalt durch die folgenden Kontextteile dargestellt wird. Verwenden Sie diese, um die Frage am Ende zu beantworten.
        Die Dateien beschäftigen sich mit Themen aus der Bauwirtschaft. Du kannst davon ausgehen, dass der Benutzer ein Bauingenieur oder Architekt ist.
        Wenn Sie die Antwort nicht wissen, sagen Sie einfach, dass Sie es nicht wissen. Versuchen Sie NICHT, sich eine Antwort auszudenken.
        Wenn die Frage nicht mit dem Kontext zusammenhängt, antworten Sie höflich, dass Sie nur Fragen beantworten, die mit dem Kontext zusammenhängen.
        Gehen Sie bei Ihrer Antwort so detailliert wie möglich vor.

        Kontext: {context}
        =========
        Frage: {question}
        ======
        """
    # english prompt
    QA_PROMPT_EN = PromptTemplate(template=qa_template_en, input_variables=["context","question" ])
    
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = self.llm_settings.get_correct_llm() # not found!!!

        retriever = self.vectors.as_retriever()


        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 