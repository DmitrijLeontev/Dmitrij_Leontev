"""
Tutorial 2. Chat with hugging face with custom context
Практика 2. Чат с языковой моделью hugging face с заданным контекстом.
"""

from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from dotenv import dotenv_values
from helpers import read_from_file, read_from_excel, read_from_excel_advanced

env = dotenv_values()

embedder_model = "sentence-transformers/all-MiniLM-L6-v2"

# =============================================================================
# ВАРИАНТ 1: Простое чтение из Excel (все колонки)
# =============================================================================
print("=== Вариант 1: Простое чтение всех колонок ===")
docs = read_from_excel('data/phones.xlsx')

# =============================================================================
# ВАРИАНТ 2: Чтение только определенных колонок
# =============================================================================
# print("=== Вариант 2: Чтение определенных колонок ===")
# docs = read_from_excel(
#     file_path='data/phones.xlsx',
#     content_columns=['brand', 'model', 'price', 'specifications'],  # Колонки для контента
#     meta_columns=['brand', 'model', 'price']  # Колонки для метаданных
# )

# =============================================================================
# ВАРИАНТ 3: Продвинутое чтение с шаблоном
# =============================================================================
# print("=== Вариант 3: Чтение с шаблоном ===")
# template = """
# Модель телефона: {brand} {model}
# Стоимость: {price} рублей
# Основные характеристики: {specifications}
# Описание: {description}
# """
# docs = read_from_excel_advanced(
#     file_path='data/phones.xlsx',
#     template=template
# )

# =============================================================================
# ВАРИАНТ 4: Чтение с другого листа
# =============================================================================
# print("=== Вариант 4: Чтение с другого листа ===")
# docs = read_from_excel(
#     file_path='data/phones.xlsx',
#     sheet_name='smartphones'  # Имя листа
# )

# Проверяем, что документы загружены
if not docs:
    print("⚠️ ВНИМАНИЕ: Не загружено ни одного документа!")
    print("Проверьте путь к файлу и структуру данных")
else:
    print(f"✅ Успешно загружено {len(docs)} документов")
    print("Пример первого документа:")
    print(docs[0].content[:200] + "...")

template = [
    ChatMessage.from_user(
        """
Ты консультант по продаже сотовых телефонов. 
Отвечай ТОЛЬКО на основе предоставленного контекста.
Если в контексте нет информации для ответа, скажи об этом.

Вопрос: {{question}}

Контекст для ответа:
{% for document in documents %}
{{ document.content }}
---
{% endfor %}

Ответ:
"""
    )
]

# Создаем document store и добавляем документы с эмбеддингами
document_store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
doc_embedder.warm_up()

if docs:
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])
    print(f"✅ Документы добавлены в хранилище с эмбеддингами")
else:
    print("❌ Нет документов для добавления в хранилище")
    # Создаем пустое хранилище
    document_store = InMemoryDocumentStore()

# Создаем эмбеддер для запросов пользователя
text_embedder = SentenceTransformersTextEmbedder(model=embedder_model)

# Создаем ретривер, чат-генератор и билдер промптов
retriever = InMemoryEmbeddingRetriever(document_store, top_k=3)
chat_generator = OllamaChatGenerator(
    model="gemma3:4b", 
    url="http://127.0.0.1:11434",
    timeout=120
)

prompt_builder = ChatPromptBuilder(template=template, required_variables=["question"])

# Создаем пайплайн и добавляем компоненты
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# Соединяем компоненты
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")


def run_pipeline(question):
    """Запуск пайплайна RAG с обработкой ошибок"""
    try:
        if not docs:
            return "❌ База знаний пуста. Пожалуйста, добавьте данные в Excel файл."
        
        response = basic_rag_pipeline.run(
            {
                "text_embedder": {"text": question}, 
                "prompt_builder": {"question": question}
            }
        )
        
        return response["llm"]["replies"][0].text
        
    except Exception as e:
        return f"❌ Ошибка при обработке запроса: {str(e)}"