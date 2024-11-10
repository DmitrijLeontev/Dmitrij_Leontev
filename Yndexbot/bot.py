import logging
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Замените эти значения на ваши собственные
OAUTH_TOKEN = 'y0_AgAAAABRx8rJAAT.....................'    #Ваш OAUTH_TOKEN
FOLDER_ID = 'b1g...........................'                                  #'<идентификатор каталога>'
API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"   


def get_iam_token():
    response = requests.post(
        'https://iam.api.cloud.yandex.net/iam/v1/tokens',
        json={'yandexPassportOauthToken': OAUTH_TOKEN}
    )
    response.raise_for_status()
    return response.json()['iamToken']


async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Привет! Я отвечу на любой ваш вопрос! Только предыдущие я забываю! Пишите подробно!')


async def process_message(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    logger.info(f'Получено сообщение от пользователя: {user_text}')

    # Получаем IAM токен
    try:
        iam_token = get_iam_token()
        logger.info(f'Получен IAM-токен: {iam_token}')
    except requests.RequestException as e:
        logger.error(f'Ошибка при получении IAM-токена: {e}')
        await update.message.reply_text('Произошла ошибка при получении токена.')
        return

    # Отправляем запрос к Yandex GPT
    data = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt",
        "completionOptions": {"temperature": 0.3, "maxTokens": 1000},
        "messages": [
            {"role": "system", "text": "отвечу на любые ваши  впоросы!"},
            {"role": "user", "text": user_text}
        ]
    }

    try:
        response = requests.post(
            API_URL,
            headers={"Accept": "application/json", "Authorization": f"Bearer {iam_token}"},
            json=data
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f'Получен ответ от Yandex GPT: {result}')
        answer = result.get('result', {}).get('alternatives', [{}])[0].get('message', {}).get('text',
                                                                                              'Ошибка получения ответа.')
    except requests.RequestException as e:
        logger.error(f'Ошибка при запросе к Yandex GPT: {e}')
        answer = 'Произошла ошибка при запросе к Yandex GPT.'

    await update.message.reply_text(answer)


def main() -> None:
    # Замените токен на свой собственный
    application = Application.builder().token("Замените на свой токен Telegram кавычки оставить").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

    application.run_polling()


if __name__ == '__main__':
    main()