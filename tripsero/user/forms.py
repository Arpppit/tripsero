from django import forms
from .models import MyModel,Question
Answer_Choices= [
    ('Adventure','To experience something exciting'),
    ('Relax','To get away from work'),
    ('Educational','To learn about other cultures and their history'),
    ('Spiritual Retreat','To admire Mother Nature'),
    ]
class MyForm(forms.ModelForm):
  class Meta:
    model = MyModel
    fields = ["fullname", "mobile_number","Age"]
    labels = {'fullname': "Name", "mobile_number": "Mobile Number","age":"Age",}

class Questionq(forms.ModelForm):
  
  A1= forms.CharField(label='Answer1', widget=forms.Select(choices=Answer_Choices))
  A2= forms.CharField(label='Answer2', widget=forms.Select(choices=Answer_Choices))
  A3= forms.CharField(label='Answer3', widget=forms.Select(choices=Answer_Choices))
  A4= forms.CharField(label='Answer4', widget=forms.Select(choices=Answer_Choices))
  class Meta:
    model = Question
    fields = ["A1", "A2","A3","A4"]
    labels = {'A1':"Answer 1", "A2":"Answer 2","A3":"Answer 3","A4":"Answer 4",}
