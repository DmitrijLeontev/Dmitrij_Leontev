from fastapi import FastAPI
from chunks import Chunk
import openai
from pydantic import BaseModel
from dotenv import load_dotenv

# инициализация индексной базы
chunk = Chunk(path_to_base="aviodatbas.txt")

# класс с типами данных параметров 
class Item(BaseModel): 
    text: str

# создаем объект приложения
app = FastAPI()

# функция обработки get запроса + декоратор 
@app.get("/")
def read_root():
    return {"message": "answer"}

# функция обработки post запроса + декоратор 
@app.post("/api/get_answer")
def get_answer(question: Item):
    answer = chunk.get_answer(query=question.text)
    return {"message": answer}