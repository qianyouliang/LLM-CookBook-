from __future__ import annotations
import streamlit as st
import os
import sys
sys.path.append("./")  # 将父目录放入系统路径中
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
import logging
from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator, validator
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file

logger = logging.getLogger(__name__)

class MistralAIEmbeddings(BaseModel, Embeddings):
    """`MistralAI Embeddings` embedding models."""

    client: Any
    """`mistralai.MistralClient`"""

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """
        实例化MistralClient为values["client"]

        Args:
            values (Dict): 包含配置信息的字典，必须包含 client 的字段.

        Returns:
            values (Dict): 包含配置信息的字典。如果环境中有mistralai库，则将返回实例化的MistralClient类；否则将报错 'ModuleNotFoundError: No module named 'mistralai''.
        """
        from mistralai.client import MistralClient
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set in the environment variables.")
        values["client"] = MistralClient(api_key=api_key)
        return values

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings(
            model="mistral-embed",
            input=[text]
        )
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.

        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in texts]

def generate_response(input_text, api_key):
    llm = ChatMistralAI(temperature=0,api_key=api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output

def get_vectordb():
    # 定义 Embeddings
    embedding = MistralAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = './db/GIS_db'
    # 加载数据库
    vectordb = FAISS.load_local("./db/GIS_db", embedding, allow_dangerous_deserialization=True)

    return vectordb

#带有历史记录的问答链
def get_chat_qa_chain(question:str ,api_key:str):
    vectordb = get_vectordb()
    llm = ChatMistralAI(temperature=0,api_key=api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str,api_key:str):
    vectordb = get_vectordb()
    llm = ChatMistralAI(temperature=0,api_key=api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                     template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 动手学大模型应用开发(GISer Liu) Mistral版本')  # 创建应用程序的标题st.title
    api_key = st.sidebar.text_input('Mistral API Key', type='password')  # 添加一个文本输入框，供用户输入其 OpenAI API 密钥
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions=["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"]
    )
    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        # 调用 respond 函数获取回答
        answer = generate_response(prompt, api_key)
        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

if __name__ == "__main__":
    main()
