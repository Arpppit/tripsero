from django.shortcuts import render
from django.http import HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .models import MyModel,Question
from .forms import MyForm,Questionq
import json
# import pathlib
from django.http import JsonResponse
from pathlib import PurePosixPath
from .attractions_recc1 import feed_input
# Create your views here.
 


def dashboard(request):
    return render(request, "users/aindex.html",{})

def index(request):
    return render(request, "users/login.html",{"username":["A","B","C"]})
    # return HttpResponse("<h2>Hello.</h2>")

@method_decorator(csrf_exempt, name='dispatch')
def survey(request):
    if request.method == "POST":
        form = Questionq(request.POST)
        if form.is_valid():
            form.save()
        #print('ouptut is :',json.loads(request.body))
        user_info = json.loads(request.body)
        #print(user_info)
        print(type(user_info['low']),type(user_info['high']),type(user_info['location']),type(user_info['days']),user_info['cat_rating'])
        recommendation = feed_input(int(user_info['low']),int(user_info['high']),user_info['location'],user_info['days'],user_info['cat_rating'])
        #recommendation = {'name': [['Rockies classic 4-day summer tour', 'Rockies classic 4-day winter tour', 'Viator exclusive: 2-day victoria and butchart gardens tour with overnight at the fairmont empress', 'Vancouver island atv tour'], ['Whistler exotic car driving experience', 'Rockies adventure 4-day summer tour', 'Vancouver to victoria whale watching with return to vancouver by seaplane', 'Gulf islands kayak and seaplane adventure'], ['Private salmon fishing charter from vancouver for up to 4 people', 'The ultimate group package deal of victoria', 'Private instagramming tour in vancouver', '3-day tenquille lake expedition']], 'image': [['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR49EmkMsAha_Pjf4aCE166RNdfM2rtJ3dRnUJav7LWST-ZZH8R4Cx-pY3tBA&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ9GDZpBwfidyh1Rv-RBXFyctStJH0ySBrxmbUnr0yE1OZzF-IJh_pUgYvkkA&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7V9AGHkBvNFmRcyxBszYPMEy5-vCwuGNnWd8TIc92w1MdaUElgLIhqmdVNg&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQgzK5Y5xTWri6RA_9p7Y_0U9fu6z0cYIL3IkN5QxQyuapBPuyxzx8kiVEMvg&s'], ['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTk7TzOb5aOPnI27MxMevSo_jtvSxJo-AwPO6zmLEGgbkAQ21Jot60lP5T_pD8&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRePOFsK_UDkj49cbmWxzK5dNDWO4i8YPWAvwxo9rMdFCMrDDNfOlRApSUkv1c&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSE9PPurYuunkMlj7ZPlH3igK6tGenDlgz-QucQ89-FpThrmpk4J9MPn7qoiQ&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-3p-1xqx1Xj2TuRXkzHRDMc6kHXaxtceQbOZK76WABpPEQaFJgPTzSEchrw&s'], ['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNu6iNwPKRJS-gro_yJrrJqmNDuj9HGUIRZROp7butnD2libJ5HcXgeUBQfP0&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS2f6gjQteAewSl1Ji5VC3cEMrVDIxyAZcjigzeUpL6rm6htA6x8xhJVahgPQ&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTD6MURK1fXT2l_JPKq8pDjFN3Rh19bpy3YmalaA9-fiatXyfpDg3MGkV6CKSU&s', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSwDhCGpT_XCrMWuu5wthcmS2sjNz58_u5UfDM3A1EWmCe0yAdvKq1_VIyWTw&s']], 'price': [['687.0', '547.0', '631.54', '627.9'], ['887.25', '708.0', '569.0', '629.0'], ['749.77', '650.0', '567.12', '532.2299804688']], 'rating': [['4.5', '4.5', '5.0', '5.0'], ['5.0', '5.0', '4.5', '5.0'], ['5.0', '5.0', '4.5', '5.0']], 'category': [['Tours & sightseeing', 'Holiday & seasonal tours', 'Recommended experiences', 'Day trips & excursions'], ['Recommended experiences', 'Outdoor activities', 'Water sports', 'Family friendly'], ['Water sports', 'Day trips & excursions', 'Tours & sightseeing', 'Multi-day & extended tours']], 'location': [['(49.2869224548,-123.1221618652)', '(49.2869224548,-123.1221618652)', '(49.2846107483,-123.1084747314)', '(49.2642745972,-124.7592926025)'], ['(49.3590698242,-123.2670211792)', '(49.2869224548,-123.1221618652)', '(49.2847518921,-123.1394805908)', '(49.2847518921,-123.1394805908)'], ['(49.2024726868,-123.1445770264)', '(38.8258628845,-76.919593811)', '(56.1303672791,-106.3467712402)', '(50.3613166809,-122.8502731323)']]}
       # print(recommendation)
        payload = {}
        payload['status']=200
        payload['content']= recommendation
        return JsonResponse(payload)
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

def to_ml_algo(request):
    if request.method == 'POST':
        pass