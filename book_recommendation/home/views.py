from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
import os
import sys


def main(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            process_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('result.html')
    else:
        form = UploadFileForm()
    return render(request, 'home/main.html', {'form': form})


def result(request):
    return render(request, 'home/result.html')


def process_uploaded_file(file):
    # save it first
    with open(os.path.join(sys.path[0], 'home', 'uploaded_file', file.name), 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
