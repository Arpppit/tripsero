from django.shortcuts import render
from django.http import HttpResponse

from .models import MyModel,Question
from .forms import MyForm,Questionq

# Create your views here.

def dashboard(request):
    return render(request, "users/aindex.html",{})

def index(request):
    return render(request, "users/login.html",{"username":["A","B","C"]})
    # return HttpResponse("<h2>Hello.</h2>")

def survey(request):
    if request.method == "POST":
        form = Questionq(request.POST)
        if form.is_valid():
            form.save()
    else:
      form = Questionq()
    return render(request, "users/survey.html",{'form': form})

def addUser(request):  
    # return HttpResponse("<h2>Here you can add remove users ! And also we display database information here </h2>")
    return render(request, "users/dashboard.html",{"username":["Aviral","Varun","Aman"]})
   
def form(request):
  if request.method == "POST":
    form = MyForm(request.POST)
    if form.is_valid():
      form.save()
  else:
      form = MyForm()
  return render(request, 'users/signup.html', {'form': form})

def questioninfo(request):
    ques = Question.objects.all()
    return render (request, "users/userinfo.html",{"que":ques})

def Ainfo(request):
    ques = Question.objects.all()
    return render (request, "users/A.html",{"que":ques})