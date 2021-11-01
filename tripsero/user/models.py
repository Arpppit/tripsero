from django.db import models

class UserInfo(models.Model):
    user_id = models.IntegerField(default=0)
    user_name = models.CharField(max_length=50,unique=True)
    First_Name = models.CharField(max_length=50)
    Last_Name = models.CharField(max_length=50)

    def __str__(self):
        return str(self.user_id)

class Login(models.Model):
    user_id = models.IntegerField(default=0)
    user_name = models.CharField(max_length=50,unique=True)
    password = models.CharField(max_length=12)

    def __str__(self):
        return str(self.user_id)

class MyModel(models.Model):
    fullname = models.CharField(max_length=200)
    mobile_number = models.IntegerField()
    Age = models.CharField(max_length=200)
    def __str__(self):
        return str(self.fullname)

class Question(models.Model):
    A1 = models.CharField(max_length=200)
    A2 = models.CharField(max_length=200)
    A3 = models.CharField(max_length=200)
    A4 = models.CharField(max_length=200)
    def __str__(self):
        return str(self.A1)