from django.shortcuts import render
from django.http import HttpResponse


def main(request):
    context = {}
    if request.method == 'GET':
        context['image'] = 'Upload your image'
    return render(request, 'home/main.html', context)


def result(request):
    return render(request, 'home/result.html')
