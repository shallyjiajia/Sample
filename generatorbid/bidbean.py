from utils.RL.test import A3C
from generatorbid.bid import Bid
from flask import current_app
import pymysql
from dbutils.pooled_db import PooledDB
from utils.datasource import DataSource
from generatorbid.bidservice import BidService

class BidBean(BidService):
    def __init__(self):
        super(BidBean,self).__init__()
        #BidService.__init__(self)
        self.appcontext = None
