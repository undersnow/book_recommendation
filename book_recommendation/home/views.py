from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadImageForm
import os
import sys
import time


def main(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            filepath = save_uploaded_file(request.FILES['image'])
            preprocess_file(filepath)
            sim_files = get_similar_files(filepath)
            return render(request, 'home/result.html', {'sim_files': sim_files})
    else:
        form = UploadImageForm()
    return render(request, 'home/main.html', {'form': form})


def save_uploaded_file(file):
    # save it first
    timestamp = str(time.time_ns())
    filepath = os.path.join(sys.path[0], 'home', 'uploaded_file', timestamp + '_' + file.name)
    with open(filepath, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return filepath


def preprocess_file(filepath):
    return filepath


def get_similar_files(filepath):
    bookname = 'bookname1'
    img_path = 'img_path1'
    URL = 'URL1'
    return [(bookname, img_path, URL)]
