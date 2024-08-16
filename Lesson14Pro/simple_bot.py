from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
from dotenv import load_dotenv
import openai
import os
import requests
import aiohttp
import json

# Функции для сохранения и загрузки данных
def save_bot_data_to_file(bot_data, filename='bot_data.json'):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(bot_data, file, ensure_ascii=False)

def load_bot_data_from_file(filename='bot_data.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Подгружаем переменные окружения
load_dotenv()

# Передаем секретные данные в переменные
TOKEN = os.environ.get("TG_TOKEN")
GPT_SECRET_KEY = os.environ.get("GPT_SECRET_KEY")

# Передаем секретный токен chatgpt
openai.api_key = GPT_SECRET_KEY

# функция для синхронного общения с chatgpt
async def get_answer(text):
    payload = {"text": text}
    response = requests.post("http://127.0.0.1:5000/api/get_answer", json=payload)
    return response.json()

# функция для асинхронного общения с сhatgpt
async def get_answer_async(text):
    payload = {"text": text}
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:5000/api/get_answer_async', json=payload) as resp:
            return await resp.json()

# функция-обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id not in context.bot_data.keys():
        context.bot_data[user_id] = {"remaining_requests": 3, "history": []}
    await update.message.reply_text('Задайте любой вопрос ChatGPT')

# функция-обработчик команды /data
async def data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_bot_data_to_file(context.bot_data)
    await update.message.reply_text('Данные сохранены')

# функция-обработчик текстовых сообщений
async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_data = context.bot_data.get(user_id, {"remaining_requests": 0, "history": []})

    if user_data["remaining_requests"] > 0:
        user_data["history"].append(update.message.text)
        user_data["history"] = user_data["history"][-5:]
        user_data["remaining_requests"] -= 1
        context.bot_data[user_id] = user_data
        save_bot_data_to_file(context.bot_data)

        first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...')
        res = await get_answer(update.message.text)
        await context.bot.edit_message_text(text=res['message'], chat_id=update.message.chat_id, message_id=first_message.message_id)
    else:
        await update.message.reply_text('Ваши запросы на сегодня исчерпаны')

# функция, которая будет запускаться раз в сутки для обновления доступных запросов
async def callback_daily(context: ContextTypes.DEFAULT_TYPE):
    if context.bot_data != {}:
        for key in context.bot_data:
            context.bot_data[key]["remaining_requests"] = 5
        save_bot_data_to_file(context.bot_data)  # Сохраняем данные в файл
        print('Запросы пользователей обновлены')
    else:
        print('Не найдено ни одного пользователя')

def main():
    application = Application.builder().token(TOKEN).build()
    application.bot_data = load_bot_data_from_file()  # Загрузка данных
    job_queue = application.job_queue
    job_queue.run_repeating(callback_daily, interval=60, first=10)
    application.add_handler(CommandHandler("start", start, block=False))
    application.add_handler(CommandHandler("data", data, block=False))
    application.add_handler(MessageHandler(filters.TEXT, text, block=False))
    application.run_polling()
    print('Бот остановлен')

if __name__ == "__main__":
    main()
