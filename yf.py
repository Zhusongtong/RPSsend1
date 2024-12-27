# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 18:45:18 2023

@author: wonder
"""

import os
import time
import numpy as np
import urllib.request
import pandas as pd
import requests
import json
import concurrent.futures
import requests
from datetime import timedelta, datetime
import time
import tushare as ts
##
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
#os.chdir(r"D:\qu\quant")
filepath = os.getcwd() + os.sep 
os.chdir(filepath)
print(filepath)
path = filepath + 'data' + os.sep 
print(path)

def dcurlapi(stock, lmt):
    # 创建logger，如果参数为空则返回root logger
    if(stock == ""):
        return
    if (stock[0:1] == "6"):
        secid = "1."+stock
    if (stock[0:1] == "0"):
        secid = "0."+stock
    if (stock[0:1] == "3"):
        secid = "0."+stock
    t = time.time()
    utc = int(round(t * 1000))
    # print(utc)
    DCK_URL_P1 = "http://55.push2his.eastmoney.com/api/qt/stock/kline/get?secid="
    DCK_URL_P2 = "&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&end=20500101&lmt="
    DCK_URL_P3 = "&_"
    url = DCK_URL_P1 + secid + DCK_URL_P2 + str(lmt) + DCK_URL_P3 + str(utc)
    return url


def geturllist(symbollist,n):
    df = pd.DataFrame()
    for i in range(len(symbollist)):
        url = dcurlapi(symbollist[i], n)
        new=pd.DataFrame({'url':url},index=[i])   # 自定义索引为：1 ，这里也可以不设置index
        df = pd.concat([new, df], ignore_index=True)
    return df

def getgz2000stock():
    stock = pd.read_csv("gz2000.csv",dtype={'Code':str})
    print(len(stock))
    print(stock.dtypes)
    colNameDict = {
    'Code':'code',
    'Name':'name'
    }
    stock.rename(columns = colNameDict,inplace=True)
    stock = stock[~stock['name'].str.contains('ST')]
    stock.head()
    return stock

def getzz500stock():
    stock = pd.read_csv("zz500.csv",dtype={'Code':str})
    colNameDict = {
    'Code':'code',
    'Name':'name'
    }
    stock.rename(columns = colNameDict,inplace=True)
    stock = stock[~stock['name'].str.contains('ST')]
    stock.head()
    return stock

#
def getcybstock():
    stock = pd.read_csv("cyb.csv",dtype={'code':str})
    stock = stock[~stock['name'].str.contains('ST')]
    print(stock.dtypes)
    return stock

def getallstock():
    stock = pd.read_csv("sw0501.csv",dtype={'code':str})
    stock = stock[~stock['name'].str.contains('ST')]
    #print("getallstock: %.2f..finished..."%len(stock))
    colNameDict = {
    'l1name':'industry1',
    'l2name':'industry2',
    'l3name':'industry3'
    }
    
    stock.rename(columns = colNameDict,inplace=True)
    return stock
 
def geths300stock():
    stock = pd.read_csv("hs300.csv",dtype={'IndexCode':str,'Code':str})
    colNameDict = {
    'Code':'code',
    'Name':'name'
    }
    stock.rename(columns = colNameDict,inplace=True)
    return stock

def getzz1000stock(): 
    stock = pd.read_csv("zz1000weight.csv",dtype={'code':str})
    stock = stock[~stock['name'].str.contains('ST')]
    return stock

# 定义请求函数
def fetch(url):
    response = requests.get(url)
    return response.text


# 并发请求不同URL
def download_all(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:  # 可以选择使用线程池或进程池 
      # 使用executor.map()方法，将请求函数和URL列表传入，得到一个生成器
      # 每次迭代生成器时，会自动调用请求函数，并传入对应的URL
      # 并发执行请求函数，返回的结果按照URL的顺序保存在results列表中 
      results = list(executor.map(fetch, urls))
    return results

def parser_item(result):
    text = json.loads(result)
    data = text['data']
    code = data['code']
    name = data['name']
    klines = data['klines']
    df=pd.DataFrame({'code':code,'name':name,'klines':klines})
    df['code'] = df['code'].astype('str')
    df_tmp = df['klines'].str.split(',',expand = True)
    df = pd.concat([df,df_tmp],axis=1)
    df.rename(columns={0:'date',1:'open',2:'close',3:'high',4:'low',5:'vol',6:'mount',7:'pamp',8:'pchg',9:'chg',10:'turnover'},inplace= True)
    df = df.drop(['klines'],axis = 1) 
    return df
    

def parser_all(text_list):
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:  # 可以选择使用线程池或进程池 
      # 使用executor.map()方法，将请求函数和URL列表传入，得到一个生成器
      # 每次迭代生成器时，会自动调用请求函数，并传入对应的URL
      # 并发执行请求函数，返回的结果按照URL的顺序保存在results列表中 
      results = executor.map(parser_item, text_list)
    merge_df = pd.concat(results,ignore_index= True) 
    return merge_df

def etl(market,day):
    time1 = time.time()
    if (market== 0):  #全部
        stock = getallstock()
    if (market== 1): ##cyb
        stock = getcybstock()
    if (market== 2):  ##zz500
        stock = getzz500stock()  
    if (market== 3):   ##gz2000
        stock = getgz2000stock()
    if (market== 4):   ##hs300
        stock = geths300stock()
    if (market== 5):   ##zz1000
        stock = getzz1000stock()
    sslist  = stock['code'].tolist()
    df_url = geturllist(sslist,day)
    urls  = df_url['url'].tolist()
    text_list = download_all(urls)
    time2 = time.time() -time1
    print(time2)
    df_hq = parser_all(text_list)
    df_hq['mount'] = df_hq['mount'].astype(float)
    df_hq['pchg'] = df_hq['pchg'].astype(float)
    df_hq['vol'] = df_hq['vol'].astype(float)
    #df_hq.to_csv('df_hq.csv',index = False) 
    time3 = time.time() -time1
    print(time3)
    return df_hq











    
