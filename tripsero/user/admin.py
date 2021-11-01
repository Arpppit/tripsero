from django.contrib import admin
from .models import UserInfo,Login,MyModel,Question
admin.site.register(Login)
admin.site.register(UserInfo)
admin.site.register(MyModel)
admin.site.register(Question)
