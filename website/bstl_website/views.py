from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render

def index(request):
  return(render(request, 'index.html'))

def about(request):
  return(render(request, 'about.html'))

def services(request):
  return(render(request, 'services.html'))

def news(request):
  return(render(request, 'news.html'))

def contact(request):
  return(render(request, 'contact.html'))
