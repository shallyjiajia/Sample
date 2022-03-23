import os
from core.xmlappcontext import XmlAppContext
from generatorbid.bidservice import TestBid

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(CURRENT_PATH)
CONFIG_PATH=os.path.join(BASE_PATH,"conf")
CONFIGFILENAME=os.path.join(CONFIG_PATH,'appcontext.xml')

context = XmlAppContext(CONFIGFILENAME)

testbid = TestBid()
testbid.context_test(context)

