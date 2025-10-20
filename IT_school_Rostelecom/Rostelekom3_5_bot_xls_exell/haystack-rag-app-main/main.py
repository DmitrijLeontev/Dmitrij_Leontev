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
