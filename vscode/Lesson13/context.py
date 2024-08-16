from telegram.ext import Application, ContextTypes, MessageHandler, filters
from telegram import Update
from dotenv import load_dotenv
import time
import os

# подгружаем переменные окружения
load_dotenv()

# токен бота
TOKEN = os.getenv('TG_TOKEN')


# функция-обработчик текстовых сообщений
async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # отправка сообщения
    self_message = await context.bot.send_message(chat_id=update.message.from_user.id, text='Привет пользователь!')

    # редактирование сообщения
    time.sleep(5)
    await context.bot.edit_message_text(chat_id=update.message.from_user.id, message_id=self_message.message_id, text='Привет юзер!')

    # закрепляем сообщение в чате
    # await context.bot.pin_chat_message(chat_id=update.message.chat_id, message_id=update.message.message_id)


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