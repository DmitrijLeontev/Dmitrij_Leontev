from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# класс с типами данных параметров 
class Item(BaseModel):
    number1: float
    number2: float

# создаем объект приложения
app = FastAPI()

# функция-обработчик post запроса для сложения
@app.post("/add")
def add_numbers(item: Item):
    return {"result": item.number1 + item.number2}

# функция-обработчик post запроса для вычитания
@app.post("/subtract")
def subtract_numbers(item: Item):
    return {"result": item.number1 - item.number2}

# функция-обработчик post запроса для умножения
@app.post("/multiply")
def multiply_numbers(item: Item):
    return {"result": item.number1 * item.number2}

# функция-обработчик post запроса для деления
@app.post("/divide")
def divide_numbers(item: Item):
    if item.number2 == 0:
        raise HTTPException(status_code=400, detail="Division by zero is not allowed")
    return {"result": item.number1 / item.number2}

