import pymysql
from dbutils.pooled_db import PooledDB
from flask import Flask, Response
from flask import current_app
import os
from core.xmlappcontext import XmlAppContext

from generatorbid.bidservice import TestBid
from utils.RL.dqn.test_dqn import Test_DQN

app = Flask(__name__)

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/bid/a3c")
def runGeneratorA3C():
    testbid = TestBid()
    output = testbid.web_test()
    return output

@app.route("/dqn")
def runDQN():
    testdqn = Test_DQN()
    output = testdqn.test()
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    #CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    #BASE_PATH = CURRENT_PATH
    #CONFIG_PATH = os.path.join(BASE_PATH, "conf")
    CONFIGFILENAME = 'conf/appcontext.xml'

    context = XmlAppContext(CONFIGFILENAME)
    with app.app_context():
        current_app.config["context"] = context

    app.run()
