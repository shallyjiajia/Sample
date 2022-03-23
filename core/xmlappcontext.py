import pymysql
from dbutils.pooled_db import PooledDB
from core.appcontext import AppContext
from core.applicationevent import ApplicationEvent
from core.applicationeventpublisher import ApplicationEventPublisher
from core.applicationlistener import ApplicationListener
from xml.dom.minidom import parse
from utils.classutil import ClassUtil
from datetime import datetime

class XmlAppContext(AppContext):
    def __init__(self,configfilename):
        self.configfilename=configfilename
        self.aep = None #aep : ApplicationEventPublisher
        self.beans = {}

        self.document = self._loadConfigurations(configfilename)

        self.refresh()

    def _loadConfigurations(self,filename):
        dom = parse(filename)
        document = dom.documentElement
        return document

    #Override
    def refresh(self):
        self._initApplicationEventPublisher()
        self._onRefresh()
        self._registerListeners()
        self._finishRefresh()

    def _initApplicationEventPublisher(self):
        self.aep = ApplicationEventPublisher()

    def _onRefresh(self):
        #database connection pool
        #Email configuration
        #Listeners
        #Context Processors
        #Bean Processors
        #Business Beans
        self._createBeans()


    def _registerListeners(self):
        self.aep.addApplicationListener(ApplicationListener())

    def _finishRefresh(self):
        self._publishEvent(ApplicationEvent("Context Refreshed",self))

    def _publishEvent(self,event:ApplicationEvent):
        self.aep.publishEvent(event)

    def _createBeans(self):
        beandefs = self.document.getElementsByTagName('bean')
        for beandef in beandefs:
            beanid = beandef.getAttribute('id')
            beancls = beandef.getAttribute('class')
            beaninitmethod = beandef.getAttribute('initmethod')
            print(beanid, beancls, beaninitmethod)

            bean = ClassUtil.newInstance(beancls)

            properties = beandef.getElementsByTagName('property')
            for property in properties:
                ptype = property.getAttribute('type')  # childNodes[0].nodeValue
                pname = property.getAttribute('name')  # childNodes[0].nodeValue
                pvalue = property.getAttribute('value')  # childNodes[0].nodeValue
                if pvalue == "": #reference
                    pvalue = property.getAttribute('ref')
                    pref = self.getBean(pvalue)
                    setattr(bean, pname, pref)
                else:
                    if ptype=="str":
                        setattr(bean, pname, pvalue)
                    elif ptype=="int":
                        setattr(bean, pname, int(pvalue))
                    elif ptype=="date":
                        setattr(bean, pname, datetime.strptime(pvalue,'%Y-%m-%d'))
                    else:
                        setattr(bean, pname, pvalue)

            bean.appcontext = self
            self.beans[beanid] = bean

            if beaninitmethod != "":
                im = getattr(bean, beaninitmethod)
                im()

    def getBean(self,beanname):
        return self.beans[beanname]