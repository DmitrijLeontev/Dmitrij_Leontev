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

