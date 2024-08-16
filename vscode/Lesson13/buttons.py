from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardMarkup, Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from dotenv import load_dotenv
import os

# подгружаем переменные окружения
load_dotenv()

# токен бота
TOKEN = os.getenv('TG_TOKEN')


# INLINE
# форма inline клавиатуры
inline_frame = [[InlineKeyboardButton("Новая задача", callback_data="task")],
                [InlineKeyboardButton("История", callback_data="history")],
                [InlineKeyboardButton("Документация", callback_data="docs", url='https://docs.python-telegram-bot.org/en/v20.6/index.html')]]
# создаем inline клавиатуру
inline_keyboard = InlineKeyboardMarkup(inline_frame)


# REPLY
# форма reply клавиатуры
reply_frame = [['Москва','Санкт-Петербург','Екатеринбург','Уфа']]
# создаем reply клавиатуру
reply_keyboard = ReplyKeyboardMarkup(reply_frame,
                                     resize_keyboard=True,    # автоматический размер кнопок
                                     one_time_keyboard=True)  # скрыть коавиатуру после нажатия


# функция-обработчик команды /start
async def start(update: Update, _):

    # прикрепляем inline клавиатуру к сообщению
    await update.message.reply_text('Пример inline клавиатуры:', reply_markup=inline_keyboard)


# функция-обработчик команды /city
async def city(update: Update, _):

    # прикрепляем reply клавиатуру к сообщению
    await update.message.reply_text('Пример reply клавиатуры:', reply_markup=reply_keyboard)


# функция-обработчик нажатий на кнопки
async def button(update: Update, _):

    # получаем callback query из update
    query = update.callback_query

    # всплывающее уведомление
    await query.answer('Это всплывающее уведомление!')
    
    # редактируем сообщение после нажатия
    await query.edit_message_text(text=f"Вы нажали на кнопку: {query.data}")


def main():

    # создаем приложение и передаем в него токен
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    # добавляем обработчик команды /city
    application.add_handler(CommandHandler("city", city))

    # добавляем CallbackQueryHandler (только для inline кнопок)
    application.add_handler(CallbackQueryHandler(button))

    # запускаем бота (нажать Ctrl-C для остановки бота)
    application.run_polling()
    print('Бот остановлен')


if __name__ == "__main__":
    main()