"""tripsero URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from . import views
urlpatterns = [
    # path('admin/', admin.site.urls),
    url(r'^$', views.index, name='index'),
    url(r'^adduser/$', views.addUser, name='addUser'),
    url(r'^form/$', views.form, name='form'),
    url(r'^survey/$', views.survey, name='survey'),
    url(r'^que/$', views.questioninfo, name='que'),
    url(r'^A/$', views.Ainfo, name='A'),
    url(r'^dashboard/$', views.dashboard, name='dashboard'),
]
