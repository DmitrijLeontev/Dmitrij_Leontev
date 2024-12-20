from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
from dotenv import load_dotenv
import openai
import os
import requests
import aiohttp
import json

# подгружаем переменные окружения
load_dotenv()

# передаем секретные данные в переменные
TOKEN = os.environ.get("TG_TOKEN")
GPT_SECRET_KEY = os.environ.get("GPT_SECRET_KEY")

# передаем секретный токен chatgpt
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
    if update.message.from_user.id not in context.bot_data.keys():
        context.bot_data[update.message.from_user.id] = 3
    await update.message.reply_text('Задайте любой вопрос ChatGPT')

# функция-обработчик команды /data 
async def data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with open('data.json', 'w') as fp:
        json.dump(context.bot_data, fp)
    await update.message.reply_text('Данные сгружены')

# функция-обработчик команды /status
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id in context.bot_data:
        remaining_requests = context.bot_data[update.message.from_user.id]
        await update.message.reply_text(f'Осталось запросов: {remaining_requests}')
    else:
        await update.message.reply_text('У вас нет активных запросов')

# функция-обработчик текстовых сообщений
async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.bot_data[update.message.from_user.id] > 0:
        first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...')
        res = await get_answer(update.message.text)
        await context.bot.edit_message_text(text=res['message'], chat_id=update.message.chat_id, message_id=first_message.message_id)
        context.bot_data[update.message.from_user.id] -= 1
    else:
        await update.message.reply_text('Ваши запросы на сегодня исчерпаны')

# функция, которая будет запускаться раз в сутки для обновления доступных запросов
async def callback_daily(context: ContextTypes.DEFAULT_TYPE):
    if context.bot_data != {}:
        for key in context.bot_data:
            context.bot_data[key] = 5
        print('Запросы пользователей обновлены')
    else:
        print('Не найдено ни одного пользователя')

def main():
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')
    job_queue = application.job_queue
    job_queue.run_repeating(callback_daily, interval=60, first=10)
    application.add_handler(CommandHandler("start", start, block=False))
    application.add_handler(CommandHandler("data", data, block=False))
    application.add_handler(CommandHandler("status", status, block=False))
    application.add_handler(MessageHandler(filters.TEXT, text, block=False))
    application.run_polling()
    print('Бот остановлен')

if __name__ == "__main__":
    main()
