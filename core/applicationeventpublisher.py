from core.applicationevent import ApplicationEvent
from core.applicationlistener import ApplicationListener

class ApplicationEventPublisher():
    def __init__(self):
        self.listeners = []

    def publishEvent(self,event:ApplicationEvent):
        for listener in self.listeners:
            listener.onApplicationEvent(event)

    def addApplicationListener(self,listener:ApplicationListener):
            self.listeners.append(listener)
