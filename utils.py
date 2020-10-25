import logging, sys, re, argparse
import json
from urllib.request import quote

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import requests
from datetime import date

def getlnglat(address):
  url = 'http://api.map.baidu.com/geocoding/v3/'
  output = 'json'
  ak = '3cCTMauuKyz68e0fSddaGYGuejwssA2T' # 百度地图ak，具体申请自行百度，提醒需要在“控制台”-“设置”-“启动服务”-“正逆地理编码”，启动
  address = quote(address) # 由于本文地址变量为中文，为防止乱码，先用quote进行编码
  uri = url + '?' + 'address=' + address + '&output=' + output + '&ak=' + ak +'&callback=showLocation '+'//GET'
  res=requests.get(uri).text
  temp = json.loads(res) # 将字符串转化为json
  m = temp['status']
  if m == 0:
    lat = temp['result']['location']['lat']
    lng = temp['result']['location']['lng']
    precise = temp['result']['precise']
    confidence = temp['result']['confidence']
    comprehension = temp['result']['comprehension']
    level = temp['result']['level']
  else:
   lat=0
   lng=0
   precise=0
   confidence=0
   comprehension=0
   level=''
  #return lat,lng,precise,confidence,comprehension,level # 纬度 latitude,经度 longitude
  return level

def get_date_info(v):
    hs={
        '凌晨':[0,6],
        '早晨':[6,8],
        '上午':[8,12],
        '中午':[12,14],
        '下午':[14,18],
        '晚上':[18,21],
        '深夜':[21,24]
    }

    ws={
        '1':'一',
        '2':'二',
        '3':'三',
        '4':'四',
        '5':'五',
        '6':'六',
        '7':'天',
    }

    v = re.sub(r'\s+', '', v)
    v = v.replace('点', '时').replace('年','-').replace('月','-').replace('日',' ')
    ymdh = re.search(r"(\d{4})\-(\d{1,2})\-(\d{1,2})\s{1}(\d{1,2})*",v)
    if ymdh is None:
      return None

    print(ymdh.groups())
    weekday = date(int(ymdh.groups()[0]),int(ymdh.groups()[1]),int(ymdh.groups()[2])).isoweekday()
    print(weekday)

    h = '凌晨'
    hour = 0
    if len(ymdh.groups())>=4 and ymdh.groups()[3] is not None:
        hour = int(ymdh.groups()[3])
    if hour>=0:
        for k,v in hs.items():
            if hour>=v[0] and hour<v[1]:
                h = k
                break

    return h,'星期'+ws[str(weekday)]

def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
