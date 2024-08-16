from django.shortcuts import render
from django.http import HttpResponse
import requests

# Create your views here.
def start(request):
    return render(request, 'start.html')

def chatbot(request):
    return render(request, 'chatbot.html')


def stats(request):
    # Здесь вы делаете запрос к FastAPI для получения статистики
    fastapi_url = 'http://127.0.0.1:5000/api/get_answer'  # Замените на фактический URL вашего FastAPI
    response = requests.post(fastapi_url, json={'text': 'dummy question'})
    
    if response.status_code == 200:
        data = response.json()
        stats_data = data.get('stats', {})
    else:
        stats_data = {}

    return render(request, 'stats.html', {'stats': stats_data})

def index(request):
    return render(request, 'index.html')
