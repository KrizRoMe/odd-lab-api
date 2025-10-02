from django.urls import path
from .views import ValueBetsView

urlpatterns = [
    path("", ValueBetsView.as_view(), name="valuebets"),
]
