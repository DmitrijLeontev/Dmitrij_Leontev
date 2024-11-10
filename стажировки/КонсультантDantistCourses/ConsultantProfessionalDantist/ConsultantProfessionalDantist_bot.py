# -*- coding: utf-8 -*-

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from openai import OpenAI
import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Загрузка необходимых библиотек
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Отключение логов httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# Подгружаем переменные окружения
TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Создаем клиент OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Параметры
model_name = "gpt-4o-mini"

path_to_base = 'baza_3.0.txt'

system_prompt = '''Ты - нейро-консультант учебного центра ООО «ПРОФЕССИОНАЛ», курсы повышения квалификации и непрерывного образования
 для врачей стоматологов в различных городах России. Предоставь точную информацию о курсах повышения квалификации и непрерывного 
 образования для врачей-стоматологов, доступных в различных городах России. При ответах на вопросы используй базу знаний с информацией
   о курсах,  данную тебе. Ничего от себя не придумывай и отвечай строго по тексту базы знаний про каждый конкретный курс. 
Будь вежливым и доброжелательным при общении.
Никогда не приветствуй пользователя!!!
Когда пользователь представляется и не задал вопрос, отвечай, что рад знакомству, прошу задайте мне любой вопрос про курсы повешения
 квалификации для врачей-стоматологов.
У каждого курса есть своя программа, свой лектор и варианты городов и дат проведения обучения, не путай их между собой. 
Особенно аккуратно обращайся с датами проведения курсов, ты должен предоставить очень точную информацию по датам. 
Ответы на вопросы клиентов должны быть точными и релевантными, строго исходя из базы знаний. 
Чтобы тебе было понятнее ориентироваться в базе знаний вот её содержание:
с #1-14 полное описание каждого курса: место, дата, лектор, программа, стоимость, ссылка на страницу с описанием и регистрацией на сайте, 
#15 перечень всех курсов - бери информацию отсюда, если пользователь задает такой или аналогичный вопросы: Какие курсы у вас есть? 
#16 ссылки на биографию - бери информацию отсюда, если пользователь просит рассказать о спикере (лекторе).
#17 информация об учебном центре - тут информация об учебном центре.
Если клиент спрашивает подробную программу курса выдай в своем ответе также соответствующую ссылку на страницу сайта учебного центра с данным курсом (имеется в информации по каждому конкретному курсу в базе знаний). Обращай особое внимание каким курсом интересуется пользователь. Запоминай и сохраняй информацию о каком курсе был вопрос от пользователя, отвечай на вопросы последовательно, когда пользователь задаёт вопрос про конкретный курс отвечай именно про запрашиваемый курс, не путай ответы про курсы между собой. Если ты запутался в вопросах пользователя и не понимаешь про какой курс тебя спрашивают, задай уточняющий вопрос: уточните пожалуйста, про какой курс вы задали вопрос?
Если пользователь меняет тему или курс, забудь предыдущие детали и сосредоточься на новом запросе.
Если пользователь спрашивает: Какие курсы у вас есть? Или расскажи обо всех курсах - предоставь весь список курсов.
На вопрос пользователя про лектора - выдавай ссылки на его биографию (опыт) из базы знаний. Для каждого курса выдавай все возможные варианты дат и мест проведения обучающих курсов. Если есть два и более курса по той же тематике, то надо рассказать обо всех. Не дублируй название курса, даты курса, преподавателя, если эта информация уже указана в вопросе пользователя. Четко фиксируй в своих ответах все варианты стоимости оплаты обучения для каждого конкретного курса, важно не путать стоимость и не придумывать от себя.
Ты не должен отвечать на вопросы, не связанные с работой учебного центра ООО «ПРОФЕССИОНАЛ». 
На такие вопросы отвечай: Извините , но я не могу ответить на вопрос не относящийся к теме курсов повышения квалификации врачей 
стоматологов. Прошу задавать мне вопросы только по теме курсов повышения квалификации врачей стоматологов. 
Если вопрос касается деятельности учебного центра ООО “Профессионал” и в базе знаний нет ответа на него - ты должен ответить: 
Для уточнения данной информации прошу обратиться в наш контактный центр:  +7 921 862-98-24 или info@profistomat.ru. 
Также ты должен выявлять потребности клиента, отвечать чётко по каждому курсу.
Не пиши в своем ответе, что информация отсутствует. 
В случае запроса пользователем конкретного курса, даты, преподавателя в конце ответа обязательно давай ссылку записаться на курс. 
Ссылка записаться на курс находится на странице каждого конкретного курса на сайте.

Наши преподаватели: https://profistomat.ru/lecturers/ - вся информация о лекторах.

ВСЕ КУРСЫ ПОВЫШЕНИЯ КВАЛИФИКАЦИИ ДОСТУПНЫЕ В УЧЕБНОМ ЦЕНТРЕ ООО "ПРОФЕССИОНАЛ"

1.«СЛОЖНЫЕ СЛУЧАИ В ЭНДОДОНТИИ. БОЛЬ В СТОМАТОЛОГИИ: ПРИЧИНА ОБРАЩЕНИЯ ПАЦИЕНТА В КЛИНИКУ»
город: Волгоград
город: Магадан
лектор: Корнетова Ирина Владимировна
2.«НЕОТЛОЖНЫЕ СОСТОЯНИЯ ДЛЯ РУКОВОДЯЩЕГО СОСТАВА КЛИНИКИ»
город: Пенза
лектор: Баранников Никита Викторович
3.«НЕОТЛОЖНЫЕ И ЭКСТРЕННЫЕ СОСТОЯНИЯ В СТОМАТОЛОГИИ»
город: Пенза
город: Сочи
лектор: Баранников Никита Викторович
4.«САНИТАРНЫЕ ПРАВИЛА И НОРМЫ В ЭФФЕКТИВНОЙ РАБОТЕ АССИСТЕНТА СТОМАТОЛОГА»
город: Мурманск
город: Сочи
лектор: Шабанова Полина Александровна
5.«МАСТЕР-КЛАСС ПО РЕСТАВРАЦИИ ЖЕВАТЕЛЬНОЙ ГРУППЫ ЗУБОВ»
город: Мурманск
лектор: Рузин Иван
6.«ОРГАНИЗАЦИЯ РАБОТЫ СТАРШЕЙ МЕДИЦИНСКОЙ СЕСТРЫ СТОМАТОЛОГИЧЕСКОЙ КЛИНИКИ»
город: Мурманск
город: Сочи
лектор: Шабанова Полина Александровна
7.«МАСТЕР-КЛАСС ПО РЕСТАВРАЦИИ ПЕРЕДНЕЙ ГРУППЫ ЗУБОВ»
город: Мурманск
город: Уфа
лектор: Рузин Иван
8.«ВОЗРАСТ РЕЗЦА»
город: Чита
лектор: Сошников Алексей Сергеевич
9.«ФУНКЦИОНАЛЬНОЙ ОРИЕНТИРОВАННЫЙ ПОДХОД В ДИАГНОСТИКЕ И ПЛАНИРОВАНИИ ЛЕЧЕНИЯ ПАТОЛОГИИ ЖЕВАТЕЛЬНОГО АППАРАТА»
город: Новосибирск
лектор: Ермошенко Роман Борисович
10.«ТОТАЛЬНЫЕ РАБОТЫ НА ИМПЛАНТАХ»
город: Новосибирск
лектор: Земан Алексей
лектор: Хасратьянц Юрий
11.«СПЛИНТ ТЕРАПИЯ. ОККЛЮЗИОННЫЕ И ФУНКЦИОНАЛЬНО ДЕЙСТВУЮЩИЕ ОРТОПЕДИЧЕСКИЕ ПРИСПОСОБЛЕНИЯ. РАЗНОВИДНОСТИ. КЛАССИФИКАЦИИ. ПОКАЗАНИЯ И АЛГОРИТМЫ РАБОТЫ ПРИ РАЗЛИЧНЫХ ВИДАХ ФУНКЦИОНАЛЬНОЙ ПАТОЛОГИИ»
город: Новосибирск
лектор: Ермошенко Роман Борисович
12.«ПРОФЕССИОНАЛЬНОЕ ВЫГОРАНИЕ СОТРУДНИКОВ СТОМАТОЛОГИЧЕСКИХ КЛИНИК»
город: Новосибирск
лектор: Немова Марина Борисовна 
13.«ПАРОДОНТОЛОГИЯ ДЛЯ ХИРУРГОВ. СВОЕВРЕМЕННАЯ ИМПЛАНТАЦИЯ»
город: Волгоград
лектор: Волкова Юлия Валерьевна
14.«УПРАВЛЕНИЕ ПОВЕДЕНИЕМ РЕБЁНКА НА СТОМАТОЛОГИЧЕСКОМ ПРИЕМЕ»
город: Волгоград
лектор: Скатова Екатерина Александровна
'''

# Загрузка и обработка базы данных
with open(path_to_base, 'r', encoding='utf-8') as file:
    document = file.read()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(document)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(md_header_splits, embeddings)

# Функция для подсчета токенов
def num_tokens_from_string(text: str, model_name: str) -> int:
        encoding = tiktoken.get_encoding(model_name)
        tokens = encoding.encode(text)
        return len(tokens)

# Проверка, что каждый фрагмент является строкой
fragments = [fragment if isinstance(fragment, str) else str(fragment) for fragment in fragments]

fragment_token_counts = [num_tokens_from_string(fragment, "cl100k_base") for fragment in fragments]

# Средняя стоимость токенов
INPUT_TOKEN_COST = 0.15 / 1000000 # USD за 1000000 входящих токенов
OUTPUT_TOKEN_COST = 0.6 / 1000000  # USD за 1000000 исходящих токенов

def calculate_cost(question_token_count: int, prompt_token_count: int, answer_token_count: int) -> float:
    """
    Функция для расчета стоимости на основе количества токенов.
    """
    question_cost = (question_token_count + prompt_token_count) * INPUT_TOKEN_COST
    answer_cost = answer_token_count * OUTPUT_TOKEN_COST
    return question_cost + answer_cost

def course_context_manager(question, course_list, course_memory):
    """
    Функция для управления контекстом курсов в диалоге.

    Args:
      question: Вопрос пользователя
      course_list: Список доступных курсов
      course_memory: Текущий запомненный курс

    Returns:
      Определенный курс, или None, если курс не найден
    """
    # Промпт для анализа вопроса и определения курса
    prompt = """
    Ты - нейро-ассистент, который помогает определить курс по вопросу пользователя. Учитывай, что вопрос может содержать частичное совпадение или упоминания о курсах из следующего списка:

    {course_list}

    Вот вопрос пользователя:
    {question}

    Пожалуйста, найди, упоминается ли какой-либо курс из списка в вопросе или есть частичное совпадение с курсом. Если курс найден, верни его точное название. Если курс не найден, верни None.
    """

    # Формируем промпт
    formatted_prompt = prompt.format(course_list="\n".join(course_list), question=question)

    # Получение ответа от модели
    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "Ты - нейро-ассистент, который помогает определить курс по вопросу пользователя. Дан вопрос пользователя и список курсов, посмотри пожалуйста и найди полные или частичные совпадения в вопросе и списке курсов. Если совпадения есть, то верни ответ на основе определенного курса. Если совпадения нет, верни None. Если текущий курс уже определен, используй его для ответа на вопрос."},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0
    )

    detected_course = completion.choices[0].message.content.strip()

    if detected_course in course_list:
        course_memory['current_course'] = detected_course
        return detected_course
    elif course_memory['current_course']:
        return course_memory['current_course']
    else:
        return None
    
def insert_newlines(text: str, max_len: int = 170) -> str:
    """
    Функция разбивает длинный текст на строки определенной максимальной длины.
    """
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) > max_len:
            lines.append(current_line)
            current_line = ""
        current_line += " " + word
    lines.append(current_line)
    return "\n".join(lines)

def answer_index(system, topic, search_index, verbose=1):
    """
    Функция для ответа на вопросы с использованием индекса и расчета стоимости.

    Args:
      system: Промпт системы
      topic: Вопрос пользователя
      search_index: Индекс базы знаний
      verbose: Флаг для вывода отладочной информации

    Returns:
      answer: Ответ на вопрос
      cost: Общая стоимость в USD
      question_token_count: Количество токенов вопроса
      prompt_token_count: Количество токенов промпта
      answer_token_count: Количество токенов ответа
    """
    # Поиск релевантных отрезков из базы знаний
    docs = search_index.similarity_search(topic, k=6)
    if verbose: print('\n ===========================================: ')

    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    if verbose: print('message_content :\n ======================================== \n', message_content)

    client = OpenAI()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Ответь на вопрос. Документ с информацией для ответа: {message_content}\n\nВопрос пользователя: \n{topic}"}
    ]

    if verbose: print('\n ===========================================: ')

    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0
    )

    answer = completion.choices[0].message.content

    # Подсчет токенов
    question_token_count = num_tokens_from_string(topic, "cl100k_base")
    prompt_token_count = num_tokens_from_string(system, "cl100k_base")
    answer_token_count = num_tokens_from_string(answer, "cl100k_base")
    cost = calculate_cost(question_token_count, prompt_token_count, answer_token_count)

    return answer, cost, question_token_count, prompt_token_count, answer_token_count

def summarize_questions(dialog):
    """
    Функция возвращает саммаризированный текст диалога.
    """
    messages = [
        {"role": "system", "content": "Ты - нейро-саммаризатор. Твоя задача - саммаризировать диалог, который тебе пришел. Если пользователь назвал свое имя, обязательно отрази его в саммаризированном диалоге"},
        {"role": "user", "content": "Саммаризируй следующий диалог консультанта и пользователя: " + " ".join(dialog)}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0,
    )

    return completion.choices[0].message.content

def answer_user_question_dialog(system, db, user_question, question_history):
    """
    Функция возвращает ответ на вопрос пользователя с расчетом стоимости.
    """
    summarized_history = ""
    if len(question_history) > 0:
        # Извлекаем только вопросы и ответы для саммаризации
        summarized_history = "Вот саммаризированный предыдущий диалог с пользователем: " + \
                              summarize_questions([q + ' ' + (a if a else '') for q, a, *_ in question_history])

    topic = summarized_history + " Актуальный вопрос пользователя: " + user_question

    # Получаем ответ и стоимость
    answer, cost, question_token_count, prompt_token_count, answer_token_count = answer_index(system, topic, db)

    question_history.append((user_question, answer if answer else '', cost, question_token_count, prompt_token_count, answer_token_count))

    # Выводим саммаризированный текст, который видит модель
    if summarized_history:
        print('****************************')
        print(summarized_history)
        print('****************************')

    print(f"Стоимость текущего вопроса и ответа: ${cost:.4f}")
    print(f"Количество токенов - Вопрос: {question_token_count}, Промпт: {prompt_token_count}, Ответ: {answer_token_count}")

    return answer, cost

def run_dialog_with_course_context(system, db, course_list):
    """
    Функция для запуска диалога с учетом контекста курсов.

    Args:
      system: Промпт системы
      db: Индекс базы знаний
      course_list: Список курсов
    """
    question_history = []
    total_cost = 0.0
    course_memory = {'current_course': None}

    while True:
        user_question = input('Пользователь: ')
        if user_question.lower() == 'stop':
            break

        # Обработка контекста курса
        current_course = course_context_manager(user_question, course_list, course_memory)
        if not current_course:
            current_course = input("Не могу точно определить курс. Уточните, пожалуйста, о каком курсе идет речь: ")

        print(f"Текущий курс: {current_course}")

        # Генерация ответа с учетом курса
        answer, cost = answer_user_question_dialog(system, db, user_question, question_history)
        total_cost += cost
        print('Консультант:', insert_newlines(answer))

    print(f"Общая стоимость диалога: ${total_cost:.2f}")
    
# Загрузка списка курсов из основной базы знаний
course_list = [
    "СЛОЖНЫЕ СЛУЧАИ В ЭНДОДОНТИИ. БОЛЬ В СТОМАТОЛОГИИ",
    "НЕОТЛОЖНЫЕ СОСТОЯНИЯ ДЛЯ РУКОВОДЯЩЕГО СОСТАВА КЛИНИКИ",
    "НЕОТЛОЖНЫЕ И ЭКСТРЕННЫЕ СОСТОЯНИЯ В СТОМАТОЛОГИИ",
    "САНИТАРНЫЕ ПРАВИЛА И НОРМЫ В ЭФФЕКТИВНОЙ РАБОТЕ АССИСТЕНТА СТОМАТОЛОГА",
    "МАСТЕР-КЛАСС ПО РЕСТАВРАЦИИ ЖЕВАТЕЛЬНОЙ ГРУППЫ ЗУБОВ",
    "ОРГАНИЗАЦИЯ РАБОТЫ СТАРШЕЙ МЕДИЦИНСКОЙ СЕСТРЫ СТОМАТОЛОГИЧЕСКОЙ КЛИНИКИ",
    "МАСТЕР-КЛАСС ПО РЕСТАВРАЦИИ ПЕРЕДНЕЙ ГРУППЫ ЗУБОВ",
    "ВОЗРАСТ РЕЗЦА",
    "ФУНКЦИОНАЛЬНОЙ ОРИЕНТИРОВАННЫЙ ПОДХОД В ДИАГНОСТИКЕ И ПЛАНИРОВАНИИ ЛЕЧЕНИЯ ПАТОЛОГИИ ЖЕВАТЕЛЬНОГО АППАРАТА",
    "ТОТАЛЬНЫЕ РАБОТЫ НА ИМПЛАНТАХ",
    "СПЛИНТ ТЕРАПИЯ. ОККЛЮЗИОННЫЕ И ФУНКЦИОНАЛЬНО ДЕЙСТВУЮЩИЕ ОРТОПЕДИЧЕСКИЕ ПРИСПОСОБЛЕНИЯ",
    "ПРОФЕССИОНАЛЬНОЕ ВЫГОРАНИЕ СОТРУДНИКОВ СТОМАТОЛОГИЧЕСКИХ КЛИНИК",
    "ПАРОДОНТОЛОГИЯ ДЛЯ ХИРУРГОВ. СВОЕВРЕМЕННАЯ ИМПЛАНТАЦИЯ",
    "УПРАВЛЕНИЕ ПОВЕДЕНИЕМ РЕБЁНКА НА СТОМАТОЛОГИЧЕСКОМ ПРИЕМЕ"
]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if 'history' in context.user_data:
        await update.message.reply_text('Здравствуйте! Продолжаем...\nКакая информация Вас интересует ?')
    else:
        await update.message.reply_text('Привет! Я нейро-консультант по курсам для стоматологов компании Профессионал. Помогу с выбором и регистрацией, представлю необходимую информацию. Чем могу быть полезен?')
        context.user_data['history'] = "\n\nКонсультант: \nПривет. Я консультант"

async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'history' not in context.user_data:
        context.user_data['history'] = "\n\nКонсультант: \nПривет. Я консультант"

    user_id = update.message.from_user.id
    first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...')
    
    answer = get_answer(update.message.text)
    
    context.user_data['history'] += '\n\nПользователь: \n' + update.message.text + '\n\nКонсультант: \n' + answer

    # Вывести содержимое переменной current_courses
    print('\n\nПользователь: \n' , update.message.text)
    print('\n\nКонсультант: \n' , answer)
   

    await first_message.edit_text(answer)

def main():
    application = Application.builder().token(TOKEN).build()
    logger.info('Бот запущен...')
    
    application.add_handler(CommandHandler("start", start, block=False))
    application.add_handler(MessageHandler(filters.TEXT, text, block=False))

    application.run_polling()
    logger.info('Бот остановлен')

if __name__ == "__main__":
    main()