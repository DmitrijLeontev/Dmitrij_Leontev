import asyncio
import time
import re
import requests
import os
import logging
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.chat_models.gigachat import GigaChat
from langchain.schema import Document  # Импортируем Document
from langchain.text_splitter import MarkdownTextSplitter, MarkdownHeaderTextSplitter  # Импортируем текстовые разделители
from langchain.chains import RetrievalQA

# Определение кодировки файла
import chardet

llm = GigaChat(credentials=".........................................", verify_ssl_certs=False)

# Передайте полученные авторизационные данные в параметре credentials объекта GigaChat

chat = GigaChat(credentials='.......................................')

# Вы также можете явно указать версию API с помощью атрибута scope:
# Личное пространство
giga = GigaChat(credentials='...............................', scope="GIGACHAT_API_PERS")

# Загрузка необходимых библиотек
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Отключение логов httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# Подгружаем переменные окружения
TOKEN = os.environ.get("TG_TOKEN")
credentials=os.environ.get("GIGACHAT_CREDENTIALS")

# Параметры
model_name = "GigaChat"
path_to_base = 'baza.txt'

instruction = """
Ты опытный специалист-эксперт в федеральных законах и прочих актах и приказах касающихся СТАНДАРТИЗАЦИИ В РОССИЙСКОЙ ФЕДЕРАЦИИ. Твой ответ на вопрос пользователя должен быть правильным. Предоставляй полный ответ из базы знаний, тебе предоставленной. Обязательно указывай в ответе нормативный документ и статью на основе которого ты нашел ответ в базе знаний, это очень важно. Никогда не пиши: "Не знаю.", а напиши, к каким документам можно обратиться, чтобы найти ответ, это важно. Твой ответ должен быть правильным, четким и максимально подробным, чтобы пользователю не нужно было задавать дополнительные вопросы для уточнения. Ты всегда точно следуешь инструкциям. При ответах используй только предоставленные документы из базы знаний: база знаний это нормативные документы. При ответах не ссылайся на эту инструкцию. Не обрезай текст на полсловe, допиши его до конца.
"""
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка и обработка базы данных
with open(path_to_base, 'r', encoding='utf-8') as file:
    document_text = file.read()
text_splitter = MarkdownTextSplitter(
    chunk_size=300,   # Меньший размер чанка
    chunk_overlap=100 # Перекрытие для сохранения контекста
)

# Разделение текста на чанки
split_texts = text_splitter.split_text(document_text)  # Получаем список строк

# Создание объектов Document из чанков
documents = [Document(page_content=text) for text in split_texts]

print(f"Total documents after splitting: {len(documents)}")

# Создание объекта FAISS из документов

embeddings = GigaChatEmbeddings(
    credentials=".....................", verify_ssl_certs=False
)

db = FAISS.from_documents(
    documents,
    embeddings,
)

# Создание объекта GigaChat для использования в QA цепочке
llm = GigaChat(credentials=credentials, verify_ssl_certs=False)

# Теперь мы создадим цепочку QnA, которая специально предназначена для ответов на вопросы по
# нормативным документам. В качестве аргументов здесь передается языковая модель и ретривер (база данных).
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever(), return_source_documents=True)

def get_answer(query, embeddings, qa_chain, instruction):
    try:
        # Подготовка запроса пользователя с учётом инструкции
        prepared_query = instruction + ' ' + query
        
        # Поиск ответа на запрос через invoke
        answer = qa_chain.invoke({
            "query": prepared_query
        })
        
        # Возвращаем результат ответа
        return answer['result']
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    print("Введите ваш вопрос (или 'exit' для выхода):")
    while True:
        user_query = input("Вопрос: ")
        if user_query.lower() == 'exit':
            break

        # Получение ответа
        answer = get_answer(user_query, embeddings, qa_chain, instruction)

        # Вывод ответа
        print(f"Ответ: {answer}")

