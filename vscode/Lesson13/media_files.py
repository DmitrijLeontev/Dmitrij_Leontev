from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram import Update
from dotenv import load_dotenv
import os

# подгружаем переменные окружения
load_dotenv()

# токен бота
TOKEN = os.getenv('TG_TOKEN')


# функция-обработчик команды /start
async def start(update: Update, _):

    # возвращаем пользователю картинку с подписью
    await update.message.reply_photo('images/bot_image.jpg', caption='Текстовое сообщение получено!')


# функция-обработчик сообщений с изображениями
async def image(update: Update, _):
    await update.message.reply_text("Фотография получена!")

    # получаем изображение из апдейта
    file = await update.message.photo[1].get_file()
    
    # сохраняем изображение на диск
    await file.download_to_drive("image.jpg")

    # сохраняем изображение как bytearray
    # img_bytearray = await file.download_as_bytearray()
    # print(img_bytearray)


# функция-обработчик голосовых сообщений
async def voice(update: Update, _):
    await update.message.reply_text("Голосовое сообщение получено!")

    # получаем файл голосового сообщения из апдейта
    new_file = await update.message.voice.get_file()

    # сохраняем голосовое сообщение на диск
    await new_file.download_to_drive('voice.mp3')


def main():

    # создаем приложение и передаем в него токен
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # добавляем обработчик сообщений с фотографиями
    application.add_handler(MessageHandler(filters.PHOTO, image))

    # добавляем обработчик голосовых сообщений
    application.add_handler(MessageHandler(filters.VOICE, voice))

    # добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    # запускаем бота (нажать Ctrl-C для остановки бота)
    application.run_polling()
    print('Бот остановлен')


if __name__ == "__main__":
    main()