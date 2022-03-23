from core.applicationevent import ApplicationEvent

class ApplicationListener():
    def __init__(self):
        pass

    def onApplicationEvent(self,event:ApplicationEvent):
        print(event.msg)