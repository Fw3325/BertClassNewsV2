import os
import sys
sys.path.append("/root/autodl-tmp/BertClassNews/")
# os.chdir('/root/autodl-tmp/BertClassNews')
print (os.getcwd(),os.listdir(),sys.path )
import config
from train.utils import Logger
log = Logger('./log/2023/app.log')
log.logger.info('yes')
# from config.config import *