# Подробная инструкция по подключению Excel файла

## 1. Создаем структуру проекта

```
rag_project/
├── data/
│   └── phones.xlsx
├── helpers.py
├── chat_pipeline.py
├── main.py
├── requirements.txt
└── .env
```

## 2. Обновленный `helpers.py` с полной реализацией для Excel

```python
from haystack import Document
import pandas as pd
import sqlite3
import psycopg2
from typing import List, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_query(query: str) -> str:
    """Format the input query for processing."""
    return query.strip().lower()

def process_results(results):
    """Process the results from the retriever to a more usable format."""
    return [result['content'] for result in results]

def log_message(message: str):
    """Log messages for debugging purposes."""
    print(f"[LOG] {message}")

def read_from_file(file_name: str) -> List[Document]:
    """Read documents from file, one line - one document"""
    try:
        with open(file_name, encoding="utf-8") as f:
            lines = f.readlines()
        return [Document(content=line.strip()) for line in lines if line.strip()]
    except Exception as e:
        logger.error(f"Error reading file {file_name}: {e}")
        return []

def read_from_excel(
    file_path: str, 
    sheet_name: str = 0,
    content_columns: Optional[List[str]] = None,
    meta_columns: Optional[List[str]] = None
) -> List[Document]:
    """
    Read documents from Excel file with flexible configuration
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name or index (default: 0 - first sheet)
        content_columns: List of columns to include in document content
        meta_columns: List of columns to include in metadata
    
    Returns:
        List of Document objects
    """
    try:
        logger.info(f"Reading Excel file: {file_path}")
        
        # Читаем Excel файл
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(f"Successfully read Excel file. Columns: {list(df.columns)}")
        logger.info(f"Number of rows: {len(df)}")
        
        # Если не указаны колонки для контента, используем все
        if content_columns is None:
            content_columns = list(df.columns)
        
        # Если не указаны колонки для метаданных, используем все
        if meta_columns is None:
            meta_columns = list(df.columns)
        
        documents = []
        
        for index, row in df.iterrows():
            try:
                # Формируем содержание документа из выбранных колонок
                content_parts = []
                for column in content_columns:
                    if column in df.columns and pd.notna(row[column]):
                        content_parts.append(f"{column}: {row[column]}")
                
                content = "\n".join(content_parts)
                
                # Формируем метаданные
                meta = {}
                for column in meta_columns:
                    if column in df.columns and pd.notna(row[column]):
                        meta[column] = row[column]
                
                # Создаем документ
                document = Document(
                    content=content, 
                    meta=meta,
                    id=f"excel_doc_{index}"  # Уникальный ID
                )
                documents.append(document)
                
                logger.debug(f"Created document {index}: {content[:100]}...")
                
            except Exception as row_error:
                logger.error(f"Error processing row {index}: {row_error}")
                continue
        
        logger.info(f"Successfully created {len(documents)} documents from Excel")
        return documents
        
    except FileNotFoundError:
        logger.error(f"Excel file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading Excel file {file_path}: {e}")
        return []

def read_from_excel_advanced(
    file_path: str,
    sheet_name: str = 0,
    template: str = None
) -> List[Document]:
    """
    Advanced Excel reader with template support
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name or index
        template: Template string for formatting content
                Example: "Модель: {brand} {model}\nЦена: {price} руб.\nХарактеристики: {specifications}"
    
    Returns:
        List of Document objects
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        documents = []
        
        for index, row in df.iterrows():
            meta = {col: row[col] for col in df.columns if pd.notna(row[col])}
            
            if template:
                # Используем шаблон для форматирования
                try:
                    content = template.format(**meta)
                except KeyError as e:
                    logger.warning(f"Missing key in template: {e}. Using default formatting.")
                    content_parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
                    content = "\n".join(content_parts)
            else:
                # Стандартное форматирование
                content_parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
                content = "\n".join(content_parts)
            
            document = Document(
                content=content,
                meta=meta,
                id=f"excel_adv_{index}"
            )
            documents.append(document)
        
        return documents
        
    except Exception as e:
        logger.error(f"Error in advanced Excel reader: {e}")
        return []

# Функции для SQL баз данных (оставлены для полноты)
def read_from_sqlite(db_path: str, table_name: str, id_column: str = "id") -> List[Document]:
    """Read documents from SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            content_parts = []
            meta = {}
            
            for i, col_name in enumerate(columns):
                if row[i] is not None:
                    content_parts.append(f"{col_name}: {row[i]}")
                    meta[col_name] = row[i]
            
            content = "\n".join(content_parts)
            document_id = meta.get(id_column, f"sqlite_{len(documents)}")
            documents.append(Document(content=content, meta=meta, id=document_id))
        
        conn.close()
        return documents
        
    except Exception as e:
        logger.error(f"Error reading SQLite database: {e}")
        return []
```

## 3. Обновленный `chat_pipeline.py` с примерами использования Excel

```python
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
```

## 4. Пример Excel файла `data/phones.xlsx`

### Структура листа:

| brand | model | price | specifications | description | in_stock | rating |
|-------|-------|-------|----------------|-------------|----------|--------|
| Xiaomi | 14 Pro | 79990 | 6.7" OLED, Snapdragon 8 Gen 3, 12GB RAM, 512GB, камера 50MP | Флагманский смартфон с лучшей камерой в линейке | yes | 4.8 |
| Samsung | Galaxy S24 Ultra | 99990 | 6.8" Dynamic AMOLED, Snapdragon 8 Gen 3, 12GB RAM, 1TB, S-Pen | Ультрафлагман с стилусом и AI-функциями | yes | 4.9 |
| Apple | iPhone 15 Pro | 104990 | 6.1" Super Retina XDR, A17 Pro, 8GB RAM, 256GB, титановый корпус | Премиальный iPhone с титановым дизайном | yes | 4.7 |
| Google | Pixel 8 Pro | 84990 | 6.7" OLED, Tensor G3, 12GB RAM, 256GB, камера 50MP | Лучшая камера среди Android-смартфонов | no | 4.6 |
| OnePlus | 12 | 69990 | 6.82" LTPO OLED, Snapdragon 8 Gen 3, 16GB RAM, 512GB, зарядка 100W | Флагман с самой быстрой зарядкой | yes | 4.5 |

## 5. Обновленный `main.py` с тестовыми запросами

```python
# Entry point for the RAG system
from pprint import pprint
from chat_pipeline import run_pipeline


if __name__ == "__main__":
    # Тестовые запросы для проверки работы с Excel
    test_queries = [
        "Сколько стоит Xiaomi 14 Pro?",
        "Какие телефоны Samsung есть в наличии?",
        "Покажи все флагманские смартфоны",
        "Какой телефон имеет лучшую камеру?",
        "Есть ли в наличии Google Pixel?",
        "Какой самый дорогой телефон?",
        "Покажи характеристики OnePlus 12"
    ]
    
    print("🚀 Запуск RAG системы с Excel базой данных\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"--- Запрос {i}: {query} ---")
        
        # Запускаем пайплайн
        response = run_pipeline(query)
        
        # Выводим ответ
        print("🤖 Ответ:")
        pprint(response)
        print("\n" + "="*50 + "\n")
```

## 6. Обновленный `requirements.txt`

```txt
haystack-ai
datasets>=2.6.1
sentence-transformers>=4.1.0
transformers[torch,sentencepiece]
python-dotenv
huggingface_hub[hf_xet]
ollama-haystack
python-telegram-bot
pandas>=1.5.0
openpyxl>=3.0.0
xlrd>=2.0.0
psycopg2-binary>=2.9.0
```

## 7. Создание Excel файла через Python (альтернатива)

Создайте файл `create_example.py` для генерации тестового Excel файла:

```python
import pandas as pd

def create_sample_excel():
    """Создание примера Excel файла с данными о телефонах"""
    
    data = {
        'brand': ['Xiaomi', 'Samsung', 'Apple', 'Google', 'OnePlus', 'Huawei', 'Realme'],
        'model': ['14 Pro', 'Galaxy S24 Ultra', 'iPhone 15 Pro', 'Pixel 8 Pro', '12', 'P60 Pro', 'GT 5 Pro'],
        'price': [79990, 99990, 104990, 84990, 69990, 74990, 45990],
        'specifications': [
            '6.7" OLED, Snapdragon 8 Gen 3, 12GB RAM, 512GB',
            '6.8" Dynamic AMOLED, Snapdragon 8 Gen 3, 12GB RAM, 1TB, S-Pen',
            '6.1" Super Retina XDR, A17 Pro, 8GB RAM, 256GB',
            '6.7" OLED, Tensor G3, 12GB RAM, 256GB',
            '6.82" LTPO OLED, Snapdragon 8 Gen 3, 16GB RAM, 512GB',
            '6.6" OLED, Kirin 9000, 8GB RAM, 256GB',
            '6.78" AMOLED, Snapdragon 8 Gen 2, 12GB RAM, 256GB'
        ],
        'description': [
            'Флагманский смартфон с лучшей камерой в линейке',
            'Ультрафлагман с стилусом и AI-функциями',
            'Премиальный iPhone с титановым дизайном',
            'Лучшая камера среди Android-смартфонов',
            'Флагман с самой быстрой зарядкой 100W',
            'Премиальный дизайн с уникальной камерой',
            'Игровой флагман с активным охлаждением'
        ],
        'in_stock': ['yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no'],
        'rating': [4.8, 4.9, 4.7, 4.6, 4.5, 4.4, 4.3],
        'category': ['flagship', 'flagship', 'flagship', 'flagship', 'flagship', 'premium', 'gaming']
    }
    
    df = pd.DataFrame(data)
    
    # Сохраняем в Excel
    with pd.ExcelWriter('data/phones.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='smartphones', index=False)
        
        # Добавляем второй лист с аксессуарами
        accessories_data = {
            'product': ['Чехол Xiaomi', 'Защитное стекло', 'Наушники', 'Зарядное устройство'],
            'price': [1990, 990, 5990, 2990],
            'compatible_with': ['Xiaomi 14 Pro', 'Все модели', 'Все телефоны', 'Все телефоны']
        }
        df_accessories = pd.DataFrame(accessories_data)
        df_accessories.to_excel(writer, sheet_name='accessories', index=False)
    
    print("✅ Пример Excel файла создан: data/phones.xlsx")
    print("📊 Основной лист: smartphones")
    print("🎧 Дополнительный лист: accessories")

if __name__ == "__main__":
    create_sample_excel()
```

## Инструкция по запуску:

1. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Создайте тестовый Excel файл:**
   ```bash
   python create_example.py
   ```

3. **Запустите систему:**
   ```bash
   python main.py
   ```

## Возможные проблемы и решения:

1. **Файл не найден** - проверьте путь `data/phones.xlsx`
2. **Ошибка чтения Excel** - установите `openpyxl` и `xlrd`
3. **Пустые документы** - проверьте структуру данных в Excel
4. **Кодировка** - убедитесь, что файл сохранен в UTF-8

Теперь у вас есть полноценная RAG система, работающая с Excel базой данных!