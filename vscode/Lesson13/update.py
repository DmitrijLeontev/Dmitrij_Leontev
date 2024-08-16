from telegram.ext import Application, MessageHandler, filters
from telegram import Update
from pprint import pprint
from dotenv import load_dotenv
import os

# подгружаем переменные окружения
load_dotenv()

# токен бота
TOKEN = os.getenv('TG_TOKEN')


# функция-обработчик текстовых сообщений
async def text(update: Update, context):
    
    # выведем в консоль содержимое update
    print(update)
    print()
    pprint(update.to_dict())
    print()

    # update.message - новое входящее сообщение любого типа - текст, фотография, наклейка и т. д.
    print(update.message.text)
    print(update.message.message_id)
    print(update.message.date)
    print(update.message.from_user.first_name)
    print(update.message.from_user.id)
    print()

    # update.callback_query - новый входящий запрос обратного вызова (используется при нажатии кнопок)
    print(update.callback_query)
    print()

    # update.update_id - уникальный id входящего сообщения
    print(update.update_id)

    await update.message.reply_text(f'Текст сообщения: {update.message.text}')


def main():

    # создаем приложение и передаем в него токен
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # добавляем обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT, text))

    # запускаем бота (нажать Ctrl-C для остановки бота)
    application.run_polling()
    print('Бот остановлен')


if __name__ == "__main__":
    main()