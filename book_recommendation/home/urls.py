from django.urls import path

from . import views

app_name = 'home'
urlpatterns = [
    path('', views.main, name='main'),
    path('result.html', views.result, name='result')
]
