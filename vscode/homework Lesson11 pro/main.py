from fastapi import FastAPI
from counter import increment_requests, get_total_requests
from chunks import Chunk
import openai
from pydantic import BaseModel  # Добавленный импорт

# Определение класса Item
class Item(BaseModel):
    text: str

# инициализация индексной базы
chunk = Chunk(path_to_base="aviodatbas.txt")

# создаем объект приложения
app = FastAPI()

# функция обработки get запроса + декоратор 
@app.get("/")
def read_root():
    increment_requests()
    return {"message": "answer"}

# функция обработки post запроса + декоратор 
@app.post("/api/get_answer")
def get_answer(question: Item):
    increment_requests()
    answer = chunk.get_answer(query=question.text)
    return {"message": answer}

# новый маршрут для получения общего количества обращений
@app.get("/total_requests")
def get_total():
    return {"total_requests": get_total_requests()}
