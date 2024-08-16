from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chunks import Chunk
from datetime import datetime
from fastapi.responses import JSONResponse

app = FastAPI()

# Настройки для работы запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Идеализация индексной базы
chunk = Chunk(path_to_base="C:/Users/Lenovo/Desktop/Django/fastapi/Simble.txt")

# Словарь для хранения статистики
request_stats = {}

# Функция для обновления статистики
def update_stats():
    current_hour = datetime.now().hour
    if current_hour not in request_stats:
        request_stats[current_hour] = 1
    else:
        request_stats[current_hour] += 1

# Класс с типами параметров данных
class Item(BaseModel):
    text: str

# Функция обработки запроса + декоратор
@app.get("/")
def read_root():
    return {"message": "ответ"}

# Функция обработки запроса + декоратор
@app.post("/api/get_answer")
def get_answer(question: Item, stats: dict = Depends(update_stats)):
    ответ = chunk.get_answer(query=question.text)
    return JSONResponse(content={"message": ответ, "stats": request_stats})
