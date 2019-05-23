from django.db import models


class Book(models.Model):
    name = models.CharField(max_length=200)
    download_url = models.URLField()
# Create your models here.
