from core.contextpostprocessor import ContextPostProcessor

class AppContext():
    def __init__(self):
        pass

    def getAppName(self):
        pass
    def getStartupDate(self):
        pass
    def addContextPostProcessor(self,processor:ContextPostProcessor):
        pass
    def refresh(self):
        pass
    def close(self):
        pass
    def isActive(self):
        pass