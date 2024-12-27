# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:33:06 2021

@author: kk
"""

import os
import json
import time
import numpy as np
import urllib.request
import pandas as pd
from datetime import timedelta, datetime
from yf  import etl
import requests
import datetime

#


##


import warnings
warnings.filterwarnings('ignore')
##


def tradeday(): 
    data = pd.read_csv("date.csv",dtype={'cal_date':str})
    datelist = data['cal_date'].tolist()
    return datelist

def isopen(date_time):
    datelist = tradeday()
    for i in datelist: 
        if(i == date_time) :
            return True
    return False

# 判断当前时间是否在（starTime~endTime）时间范围内
def checktime(startTime, endTime):
    start_time = datetime.datetime.strptime(str(datetime.datetime.now().date()) + startTime, '%Y-%m-%d%H:%M')
    end_time = datetime.datetime.strptime(str(datetime.datetime.now().date()) + endTime, '%Y-%m-%d%H:%M')
    now_time = datetime.datetime.now()
    if start_time < now_time < end_time:
        return True
    return False




##
def dcindexapi(index_code,lmt):
    #创建logger，如果参数为空则返回root logger
    if (index_code[0:1] =="B"):
        secid ="90."+index_code
    if (index_code[0:1] =="0"):
        secid ="1."+index_code
    t = time.time()
    utc = int(round(t * 1000))
    #print(utc)
    DCK_URL_P1 ="http://84.push2his.eastmoney.com/api/qt/stock/kline/get?secid=";
    DCK_URL_P2 ="&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&end=20500101&lmt="
    DCK_URL_P3 ="&_"
    url = DCK_URL_P1 + secid + DCK_URL_P2 + str(lmt) + DCK_URL_P3 + str(utc)
    #print(url)
    ##
    df=pd.DataFrame()
    re = urllib.request.urlopen(url)
    rsp = re.read()
    rsp = rsp.decode("utf-8")
    text = json.loads(rsp)
    data = text['data']
    code = data['code']
    name = data['name']
    klines = data['klines']
    #updatetime = datetime.now().strftime( "%Y-%m-%d %H:%M:%S" )
    updatetime = time.strftime("%Y-%m-%d, %H:%M:%S")
    for i in range(len(klines)):
        kine = klines[i]
        strlist = kine.split(',')
        new=pd.DataFrame({'date':strlist[0],
                          'open':strlist[1],
                          'close':strlist[2],
                          'high':strlist[3],
                          'low':strlist[4],
                          'vol':strlist[5],  ##成交量
                          'mount':strlist[6],  ##成交额
                          'pamp':strlist[7],        ##振幅
                          'pchg':strlist[8],
                          'chg':strlist[9],
                          'turnover':strlist[10],
                          'code':code,
                          'name':name,
                          'updatetime':updatetime},index=[i])   # 自定义索引为：1 ，这里也可以不设置index
        df = pd.concat([new, df], ignore_index=True)

    #print(len(df))
   
    return df
##
 


###
def boll(data, n):
    data = data.sort_values(by='date',ascending=True)
    ma = pd.Series(np.round(data['close'].rolling(n).mean(), 2), name='mb')  # 计算nday均线
    # pandas.std() 默认是除以n-1 的，即是无偏的，如果想和numpy.std() 一样有偏，需要加上参数ddof=0
    # 此处添加ddof的原因是wind和yahoo的计算均采用的有偏值进行的计算
    std = pd.Series(np.round(data['close'].rolling(n).std(ddof=0), 2))  # 计算nday标准差，有偏
    # 此处的2就是Standard Deviations
    up = pd.Series(np.round(ma + (2 * std),2), name='up')
    data = data.join(ma)  # 上边不写name 这里报错
    data = data.join(up)
    dn = pd.Series(np.round(ma - (2 * std),2), name='dn')
    data = data.join(dn)
    return data

def boll2(data, n):
    data = data.sort_values(by='date',ascending=True)
    ma = pd.Series(np.round(data['avgp'].rolling(n).mean(), 2), name='mb1')  # 计算nday均线
  #
    std = pd.Series(np.round(data['avgp'].rolling(n).std(ddof=0), 2))  # 计算nday标准差，有偏
    # 此处的2就是Standard Deviations
    up = pd.Series(np.round(ma + (2 * std),2), name='up1')
    data = data.join(ma)  # 
    data = data.join(up)
    dn = pd.Series(np.round(ma - (2 * std),2), name='dn1')
    data = data.join(dn)
    return data
##
def ma(data, n):
    data = data.sort_values(by='date',ascending=True)
    ma = pd.Series(np.round(data['close'].rolling(n).mean(), 2), name='ma%s'%n)  # 计算nday均线
    data = data.join(ma)  # 上边不写name 这里报错
    return data

def sma(data, n):
    data = data.sort_values(by='date',ascending=True)
    sma = pd.Series(np.round(data['avgp'].rolling(n).mean(), 2), name='sma%s'%n)  # 计算nday均线
    data = data.join(sma)  # 上边不写name 这里报错
    return data

def avgp(data):
    avgp = pd.Series(np.round(((data['high'].astype(float)+data['low'].astype(float)+data['close'].astype(float))/3), 2), name='avgp')  
    data = data.join(avgp)  
    return data

def avg(data):
    avg = pd.Series(np.round(data['mount'].astype(float)/data['vol'].astype(float)/100, 2), name='avg')  
    data = data.join(avg)  
    return data

##rps120 半年
def pct120(df):
    df['closemeta'] = df['close'].shift(120)
    df = df.sort_values(by='date',ascending=True)
    pct120 = pd.Series(np.round((df['close'].astype(float)- df['closemeta'].astype(float))/df['closemeta'].astype(float), 4), name='pct120')  
    df = df.join(pct120)  
    return df


def rps120(df):
    df = df.sort_values(by='pct120',ascending=False)
    df['range']=  range(1,len(df)+1)
    df['rps120'] = (1- df['range']/len(df))*100
    return df

##rps60 季度
def pct60(df):
    df['closemeta'] = df['close'].shift(60)
    df = df.sort_values(by='date',ascending=True)
    pct60 = pd.Series(np.round((df['close'].astype(float)- df['closemeta'].astype(float))/df['closemeta'].astype(float), 4), name='pct60')  
    df = df.join(pct60)  
    return df


def rps60(df):
    df = df.sort_values(by='pct60',ascending=False)
    df['range']=  range(1,len(df)+1)
    df['rps60'] = (1- df['range']/len(df))*100
    return df

##rps20 月度

def pct20(df):
    df['closemeta'] = df['close'].shift(20)
    df = df.sort_values(by='date',ascending=True)
    pct20 = pd.Series(np.round((df['close'].astype(float)- df['closemeta'].astype(float))/df['closemeta'].astype(float), 4), name='pct20')  
    df = df.join(pct20)  
    return df

def rps20(df):
    df = df.sort_values(by='pct20',ascending=False)
    df['range']=  range(1,len(df)+1)
    df['rps20'] = (1- df['range']/len(df))*100
    return df

##rps5 周度
def pct5(df):
    df['closemeta'] = df['close'].shift(5)
    df = df.sort_values(by='date',ascending=True)
    pct5 = pd.Series(np.round((df['close'].astype(float)- df['closemeta'].astype(float))/df['closemeta'].astype(float), 4), name='pct5')  
    df = df.join(pct5)  
    return df

def rps5(df):
    df = df.sort_values(by='pct5',ascending=False)
    df['range']=  range(1,len(df)+1)
    df['rps5'] = (1- df['range']/len(df))*100
    return df

def rps5chg(df):
    df = df.sort_values(by='date',ascending=True)
    df['rps5meta']  = df['rps5'].shift(1)
    rps5chg = pd.Series(np.round((df['rps5'].astype(float)- df['rps5meta'].astype(float))/df['rps5meta'].astype(float), 4), name='rps5chg')  
    df = df.join(rps5chg)  
    #df = df.join(rps5meta)  
    return df

def backtest(df):
    df = df.sort_values(by='date',ascending=True)
    df['id'] = range(len(df))
    pchglist = df['pchg'].tolist()
    closelist = df['close'].tolist()
    accretdict = {}  
    for i in range(len(df)):
        if(i == 0):
            accretdict[i] = pchglist[0]
        if(i > 0):
           accretdict[i]  = np.round(((float(closelist[i]) - float(closelist[0]))/float(closelist[0]))*100,2)
    sk_acc_ret = pd.DataFrame.from_dict(accretdict,orient='index',columns=['sk_acc_ret'])
    sk_acc_ret = sk_acc_ret.reset_index().rename(columns={'index':'id'})
    df = pd.merge(df, sk_acc_ret,on='id')
    df.drop('id',axis=1, inplace=True)
    return df  




def modelma(data):
    data = data.sort_values(by='date',ascending=True)
    sig1 = pd.Series(np.round(data['ma3'] - data['ma5'], 2), name='sig1')  
    sig2 = pd.Series(np.round(data['close'].astype(float) - data['open'].astype(float), 2), name='sig2')  
    sig1na = pd.Series(np.isnan(sig1), name='sig1na')    
    bs1 = pd.Series(sig1 >=0 , name='bs1') 
    kred = pd.Series(sig2 >=0, name='kred')
    data = data.join(bs1)  
    data = data.join(sig1na)  
    data = data.join(sig1)  
    data = data.join(kred)
    #
    return data

def maclose(df):
    df = df.sort_values(by='date',ascending=True)
    df['deltama30'] = df['ma30'] * 0.95
    ma30flag = pd.Series((df['close'].astype(float) < df['ma30'])&(df['close'].astype(float) >= df['deltama30']), name='ma30flag')
    df = df.join(ma30flag)
    ##
    df['deltama60'] = df['ma60'] * 0.95
    ma60flag = pd.Series((df['close'].astype(float) < df['ma60'])&(df['close'].astype(float) >= df['deltama60']), name='ma60flag')
    df = df.join(ma60flag)
    ##
    df['deltama120'] = df['ma120'] * 0.95
    ma120flag = pd.Series((df['close'].astype(float) < df['ma120'])&(df['close'].astype(float) >= df['deltama120']), name='ma120flag')
    df = df.join(ma120flag)
    ##
    macloseflag = pd.Series(ma30flag | ma60flag | ma120flag, name='macloseflag')
    df = df.join(macloseflag)
    return df


##下跌中，DK前一天符合MA
def strategy_beta(df):
    df = df.sort_values(by='date',ascending=True)   
    #0真阳线，收红，收盘大于0.0
    truekred = pd.Series(((~df['sig1na']) & df['kred'] & (df['pchg'].astype(float) >= 0.009)  ), name='truekred')
    df = df.join(truekred)
    #1假阴线alpha,收阴，收盘-2.49 ~ -2.99,ma3 > ma5
    greenfake = pd.Series((~df['sig1na'])  & (~ df['kred'] )& (df['pchg'].astype(float) < -2.49) & (df['pchg'].astype(float) >= -2.99) & df['bs1'] , name='greenfake')
    df = df.join(greenfake)
    #2假阴线beta,收阴，收盘大于-2.49，ma3 > ma5
    greenfake2 = pd.Series((~df['sig1na'])  & (~ df['kred'] )& (df['pchg'].astype(float) >= -2.49) & df['bs1'] , name='greenfake2')
    df = df.join(greenfake2)
    #3假阳线,收红，收盘小于-2.99
    redfake = pd.Series(((~df['sig1na']) & df['kred'] & (df['pchg'].astype(float) < -2.99)  ), name='redfake')
    df = df.join(redfake)
    #4高开低走，收绿，开盘大于等于最高，收盘小于2.99，
    openmax = pd.Series(((df['open'].astype(float) - df['high'].astype(float) )>= 0.0) & (df['pchg'].astype(float) <= 2.99)& (~df['kred'] ), name='openmax')
    df = df.join(openmax)
    #5.做多
    #5.1 收红、不是假阳、m3>m5,不是高开低走
    #5.2 假阴线beta，前一天真阳线、不是假阳、m3>m5,不是高开低走
    #5.3 假阴线alpha，不是假阳、m3>m5,不是高开低走
    #5.4 真阳、MA30，60，120突破，成交量放大1.2
    dk1 = pd.Series(((df['kred'] |
                     df['greenfake2']|
                     (df['greenfake'] & df['truekred'].shift(1))) & ~df['redfake']  & df['bs1'] & (~df['openmax'])) |
                    (df['truekred'] & df['makpos'] & (df['votratio'].astype(float) >=1.2))  
                    , name='dk1') 
    df = df.join(dk1)
    df['dkmeta'] = df['dk1'].shift(1)
    ##
    ##5.5 前一天做多，当天收盘大于0.99
    dk2 = pd.Series((df['dkmeta'] & (df['pchg'].astype(float) > 0.99)) | df['dk1'], name='dk2') 
    df = df.join(dk2)
    #6.做空
    #6.1 位于MA30、60、120、180均线下方时0.95~0.999，当天不是真阳线，做空
    #6.2 当天下穿 MA30、60、120、180均线时，做空
    dk = pd.Series( ((df['macloseflag'] & df['truekred'] & df['dk2'])| 
                     ( ~df['macloseflag']) & df['dk2'] ) &
                   ~df['makneg'],
                   name='dk') 
    df = df.join(dk)
    return df

###MA30策略
def modelma30(df):
    df = df.sort_values(by='date',ascending=True)
      #ma30突破
    df['ma30meta']  = df['ma30'].shift(1)
    df['closemeta']  = df['close'].shift(1)
    ma30pos = pd.Series((df['closemeta'].astype(float)  <= df['ma30meta'].astype(float)) & (df['close'].astype(float)  >= df['ma30'].astype(float)), name='ma30pos')
    df = df.join(ma30pos)
    ma30neg = pd.Series((df['closemeta'].astype(float)  >= df['ma30meta'].astype(float)) & (df['close'].astype(float)  <= df['ma30'].astype(float)), name='ma30neg')
    df = df.join(ma30neg)
    df.drop('closemeta',axis=1, inplace=True)
    return df

###MA60策略
def modelma60(df):
    df = df.sort_values(by='date',ascending=True)
      #ma60突破
    df['ma60meta']  = df['ma60'].shift(1)
    df['closemeta']  = df['close'].shift(1)
    ma60pos = pd.Series((df['closemeta'].astype(float)  <= df['ma60meta'].astype(float)) & (df['close'].astype(float)  >= df['ma60'].astype(float)), name='ma60pos')
    df = df.join(ma60pos)
    ma60neg = pd.Series((df['closemeta'].astype(float)  >= df['ma60meta'].astype(float)) & (df['close'].astype(float)  <= df['ma60'].astype(float)), name='ma60neg')
    df = df.join(ma60neg)
    df.drop('closemeta',axis=1, inplace=True)
    return df

###MA120策略
def modelma120(df):
    df = df.sort_values(by='date',ascending=True)
      #ma120突破
    df['ma120meta']  = df['ma120'].shift(1)
    df['closemeta']  = df['close'].shift(1)
    ma120pos = pd.Series((df['closemeta'].astype(float)  <= df['ma120meta'].astype(float)) & (df['close'].astype(float)  >= df['ma120'].astype(float)), name='ma120pos')
    df = df.join(ma120pos)
    ma120neg = pd.Series((df['closemeta'].astype(float)  >= df['ma120meta'].astype(float)) & (df['close'].astype(float)  <= df['ma120'].astype(float)), name='ma120neg')
    df = df.join(ma120neg)
    df.drop('closemeta',axis=1, inplace=True)
    return df


###MA突破策略
def modelmak(df):
    df = df.sort_values(by='date',ascending=True)
      #ma突破
    makpos = pd.Series(df['ma30pos'] | df['ma60pos'] |  df['ma120pos'], name='makpos')
    df = df.join(makpos)
     #ma回落
    makneg = pd.Series(df['ma30neg'] | df['ma60neg'] |  df['ma120neg'], name='makneg')
    df = df.join(makneg)
    return df



###成交量比率
def votratio(df):
    df = df.sort_values(by='date',ascending=True)
      #成交量比率vol
    df['volmeta']  = df['vol'].shift(1)
    votratio = pd.Series(np.round(df['vol'].astype(float) / df['volmeta'].astype(float),2) , name='votratio')
    df = df.join(votratio)
    df.drop('volmeta',axis=1, inplace=True)
    return df


###成交额比率
def vatratio(df):
    df = df.sort_values(by='date',ascending=True)
      #成交量比率vol
    df['mountmeta']  = df['mount'].shift(1)
    vatratio = pd.Series(np.round(df['mount'].astype(float) / df['mountmeta'].astype(float),2), name='vatratio')
    df = df.join(vatratio)
    df.drop('mountmeta',axis=1, inplace=True)
    return df

def obvtrend(df):
    data = df.sort_values(by='date',ascending=True)
    data['close'] = data['close'].astype(float)
    data['vol'] = data['vol'].astype(float)
    # 计算OBV
    data['OBV'] = (data['close'].diff().ge(0).map({True:1, False:-1}) * data['vol']).cumsum()
    # 计算OBV的变化趋势
    data['OBV_trend'] = data['OBV'].diff()

    # 将OBV趋势转换为'Up'（上升）和'Down'（下降）
    data['OBV_trend_direction'] = data['OBV_trend'].apply(lambda x: 'Up' if x > 0 else 'Down' if x < 0 else 'Flat')

    # 找出OBV趋势方向改变的点
    data['OBV_trend_change'] = data['OBV_trend_direction'].ne(data['OBV_trend_direction'].shift())

    # 根据趋势变化点，为每个连续的趋势区间赋予一个唯一的ID
    data['OBV_trend_id'] = data['OBV_trend_change'].cumsum()

    # 计算每个趋势区间的长度
    data['OBV_trend_length'] = data.groupby('OBV_trend_id').cumcount() + 1
    return data   
    



##
def calcop(df):
    df = df.sort_values(by='date',ascending=True)
    df['op']="空仓"
    df['dkmeta'] =df['dk'].shift(1) 
    df.loc[(df['dk']== False) & (df['dkmeta']!= True), 'op'] = '空仓'
    df.loc[(df['dk']== False) & (df['dkmeta']== True) , 'op'] = '卖出'
    df.loc[(df['dk']== True) & (df['dkmeta'] == True) , 'op'] = '持仓'
    df.loc[(df['dk']== True) & (df['dkmeta']!= True) , 'op'] = '买入'
    df.drop('dkmeta',axis=1, inplace=True)
    return df


##
def calckeep(df):
    df = df.sort_values(by='date',ascending=True)
    df['id'] = range(len(df))
    oplist = df['op'].tolist()
    kpdict = {}
    
    for i in range(len(df)):
        if(i == 0):
            kpdict[i] = 1
        if(i > 0):
            if(oplist[i] == oplist[i-1]):
                kpdict[i] = kpdict[i-1] + 1
            elif(oplist[i]=='买入'): 
                kpdict[i] = 1
            elif(oplist[i]=='卖出'):
                 kpdict[i]= 1
            elif((oplist[i]=='持仓') &( oplist[i-1]=='买入')):
                kpdict[i] = kpdict[i-1] + 1
            elif((oplist[i]=='空仓') &( oplist[i-1]=='卖出')):
                kpdict[i] = kpdict[i-1] + 1
    #print(kpdict)
    keep = pd.DataFrame.from_dict(kpdict,orient='index',columns=['keep'])
    keep = keep.reset_index().rename(columns={'index':'id'})
    #print(keep)
    df = pd.merge(df, keep,on='id')
    df.drop('id',axis=1, inplace=True)
    #df.to_csv('0314.csv',index = False) 
    return df

def calcportfolio(df):
    df = df.sort_values(by='date',ascending=True)
    df['id'] = range(len(df))
    oplist = df['op'].tolist()
    closelist = df['close'].tolist()
    retdict = {}
    accretdict = {}  
    for i in range(len(df)):
        if(i == 0):
            retdict[i] = 0.0
            accretdict[i] = 1.0
        if(i > 0):
            if((oplist[i] == "空仓")|(oplist[i]=='买入')):
               retdict[i] =  0.0
               accretdict[i] = accretdict[i-1]
            elif((oplist[i] == "持仓")|(oplist[i]=='卖出')):
                 retdict[i] = np.round(( (float(closelist[i]) - float(closelist[i-1]))/float(closelist[i-1]))*100,3)
                 accretdict[i] = np.round(float(accretdict[i-1]) * (1.0+ (float(closelist[i]) - float(closelist[i-1]))/float(closelist[i-1])),3)
    capital_ret = pd.DataFrame.from_dict(retdict,orient='index',columns=['capital_ret'])
    capital_acc_ret = pd.DataFrame.from_dict(accretdict,orient='index',columns=['capital_acc_ret'])
    capital_ret = capital_ret.reset_index().rename(columns={'index':'id'})
    capital_acc_ret = capital_acc_ret.reset_index().rename(columns={'index':'id'})
    df = pd.merge(df, capital_ret,on='id')
    df = pd.merge(df, capital_acc_ret,on='id')
    df.drop('id',axis=1, inplace=True)
    return df    
    
def cutdf(df,n):
    hs =  dcindexapi('000300',n)
    hs = hs.sort_values(by='date',ascending=True)
    df_head = hs.head(1)
    df_tail = hs.tail(1)
    start = df_head['date'].iloc[0]
    end = df_tail['date'].iloc[0]
    print(start)
    print(end)
    df = df[ (df['date'] >= start) & (df['date'] <= end)]
    return df
    

def predata(data,n):
    time1 = time.time()
    #df = data.groupby('code', as_index=False).apply(avg)
    #df = data.groupby('code', as_index=False).apply(avgp)
    #df = df.groupby('code', as_index=False).apply(boll,n=20)
    #df = df.groupby('code', as_index=False).apply(boll2,n=20)
    df = data.groupby('code', as_index=False).apply(ma,n=3)
    df = df.groupby('code', as_index=False).apply(ma,n=5)
    df = df.groupby('code', as_index=False).apply(ma,n=10)
    df = df.groupby('code', as_index=False).apply(ma,n=30)
    df = df.groupby('code', as_index=False).apply(ma,n=60)
    df = df.groupby('code', as_index=False).apply(ma,n=120)
    df = df.groupby('code', as_index=False).apply(modelma30)
    df = df.groupby('code', as_index=False).apply(modelma60)
    df = df.groupby('code', as_index=False).apply(modelma120)
    df = df.groupby('code', as_index=False).apply(votratio)
    df = df.groupby('code', as_index=False).apply(vatratio)      
    df = df.groupby('code', as_index=False).apply(maclose)
    df = df.groupby('code', as_index=False).apply(modelmak)
    ##pct120(df)
    df = df.groupby('code', as_index=False).apply(pct120)
    df = df.groupby('date', as_index=False).apply(rps120)
    ##pct60(df)
    df = df.groupby('code', as_index=False).apply(pct60)
    df = df.groupby('date', as_index=False).apply(rps60)
    ##pct20(df)
    df = df.groupby('code', as_index=False).apply(pct20)
    df = df.groupby('date', as_index=False).apply(rps20)
    ##
    ##pct5(df)
    df = df.groupby('code', as_index=False).apply(pct5)
    df = df.groupby('date', as_index=False).apply(rps5)
    #df = df.groupby('code', as_index=False).apply(rps5chg)
    ##
    ##
    df = df.groupby('code', as_index=False).apply(modelma)
    df = cutdf(df,n)
    df = df.groupby('code', as_index=False).apply(strategy_beta)  
    time2 = time.time() -time1
    return df

def getallstock():
    stock = pd.read_csv("sw0501.csv",dtype={'code':str})
    #print("getallstock: %.2f..finished..."%len(stock))
    colNameDict = {
    'l1name':'industry1',
    'l2name':'industry2',
    'l3name':'industry3'
    }
    
    stock.rename(columns = colNameDict,inplace=True)
    return stock

#df = stockquant(code,30)  
 
## market: 1,沪深300
## market: 2,创业板
## market: 3,上证、深圳全部
## market: 4,国有资本
## market: 5,中证1000
## market: 6,A50
def patquant(market,n): 
    m = n
    time1 = time.time()
    #df = pd.read_csv("df_hq.csv",dtype={'code':str})
    df = etl(market,n)
    df = predata(df,n)
    df.to_csv('predata.csv',index = False)  
    #print(df.dtypes)
    col_n = ['date','open','close','high','low','pamp','pchg','vol','mount','code','name','pct120','rps120','pct60','rps60','pct20','rps20','pct5','rps5','rps5meta','rps5chg','dk','madk','ma30pos','ma30neg','ma60pos','ma60neg','ma120pos','ma120neg','makpos','makneg','votratio','vatratio']
    df = pd.DataFrame(df,columns = col_n)
    df = df.groupby('code', as_index=False).apply(calcop)
    df = df.groupby('code', as_index=False).apply(calckeep)
    df = df.groupby('code', as_index=False).apply(calcportfolio)
     ##补充标签
    df_stock = getallstock()
    col_n = ['code','market','industry1','industry2','industry3']
    df_stock = pd.DataFrame(df_stock,columns = col_n)
    df = pd.merge(df, df_stock,on='code')
    col_n = ['date','open','close','high','low','pamp','pchg','vol','mount','code','name','market','industry1','industry2','industry3','pct120','rps120','pct60','rps60','pct20','rps20','pct5','rps5','op','keep','capital_ret','capital_acc_ret']
    df = pd.DataFrame(df,columns = col_n)
    time2 = time.time() -time1
    return df

etf_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/a332e317-b54d-462f-b724-a27ac168ae0d'

def rpspushfs(context):
   
    ##
    url = etf_url
    headers = {
            'Content-Type': 'application/json'
            }
    ##
    payload_message = {
            "msg_type": "text",
            "content": {
                    "text": context
                    }
            }

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload_message))
    return response.text
def calcrpstxt(df):
    df = df.sort_values(by='pchg',ascending=False)
    codelist = df['code'].tolist()
    namelist = df['name'].tolist()
    pchglist = df['pchg'].tolist()
    mountlist = df['mount'].tolist()
    keeplist = df['keep'].tolist()
    industry3list = df['industry3'].tolist()
    date_time = time.strftime("%Y-%m-%d, %H:%M:%S")
    trade = date_time +'\n'+'代码'+'  '+'名称'+'  '+'keep'+'  '+'涨跌幅'+'   '+'成交额'+ '  '+'行业'+'\n'
    for i in range(len(df)):
            temp =str(codelist[i])+'  '+ str(namelist[i])  +'  '+ str(keeplist[i])+'  ' + str(pchglist[i])+'%'+'  ' +str((mountlist[i]//10000)) +'万'+'  ' + str(industry3list[i])  + '\n'
            trade = trade + temp
    return trade

def calcindustrytxt(df):
    industrylist = df['industry'].tolist()
    numlist = df['num'].tolist()
    trade ='次数'+' '+' '+'行业'+'\n'
    for i in range(len(df)):
            temp =str(numlist[i])+'  '+ str(industrylist[i]) + '\n'
            trade = trade + temp
    return trade

def pushindustry(df):
    tj1 = df['industry1'].value_counts()
    s = tj1.to_frame()
    s= s.reset_index()
    new_col = ['industry', 'num']
    s.columns = new_col
    industry1text = calcindustrytxt(s)
    rpspushfs(industry1text)
    ###
    tj3 = df['industry3'].value_counts()
    s3 = tj3.to_frame()
    s3= s3.reset_index()
    new_col = ['industry', 'num']
    s3.columns = new_col
    industry3text = calcindustrytxt(s3)
    rpspushfs(industry3text)

if __name__ == '__main__':
    market = 2
    day = 130
    time1 = time.time()
    df = patquant(market,day)
    df = df[~df['name'].str.contains('ST')]
    # In[*]
    df.to_csv('1013.csv',index = False)
    # In[*]
    today ='2024-12-27'
    #today = time.strftime("%Y-%m-%d")
    result2 = df[ (df['date'] ==today) &( ( df['op'] =='买入')|( df['op'] =='持仓') )]
    # In[*]
    result2f = df[ (df['date'] ==today) &( ( df['op'] =='卖出')|( df['op'] =='空仓') )]
    result3 = df[ (df['date'] ==today) &( ( df['op'] =='买入')|( df['op'] =='持仓')  ) & (df['rps120'] > 90) & (df['rps5'] > 90) & (df['rps60'] > 95) & (df['rps20'] > 95)  ] 
    result4 = df[ (df['date'] ==today) &( ( df['op'] =='买入')|( df['op'] =='持仓')  ) & (df['rps120'] > 95) & (df['rps5'] > 95) & (df['rps60'] > 95) & (df['rps20'] > 95)  ] 
    result4a = df[ (df['date'] ==today)&(df['market'] =='创业板') &( ( df['op'] =='买入')|( df['op'] =='持仓')  ) & (df['rps120'] > 95) & (df['rps5'] > 98) & (df['rps60'] > 95) & (df['rps20'] > 95)  ]  
    result5 = df[ (df['date'] ==today)  & (df['rps120']<10) & (df['rps5'] > 90)&( ( df['op'] =='买入')|( df['op'] =='持仓') )  ] 
    result_max = df[(df['pchg'].astype(float)   >3.3) &(df['date'] ==today) &( ( df['op'] =='空仓')|( df['op'] =='卖出')  )  ]
    result_min = df[(df['pchg'].astype(float) < -5.49) &(df['date'] ==today)  ]
    result6 = df[ (df['date'] ==today) & (df['pchg'].astype(float)   > 3)  & (df['keep'] >3) & (df['rps5'] > 85)&( ( df['op'] =='买入')|( df['op'] =='持仓') )  ]  
    result666 = df[ (df['date'] ==today) &( ( df['op'] =='买入')|( df['op'] =='持仓')  ) &  (df['rps5'] > 95)  & (df['rps20'] > 95)  ] 
    result_max22 = df[(df['pchg'].astype(float) >8) &(df['date'] ==today)]
    ##
    mr =''
    if (market== 0):  #全部
        mr = '沪深全A'
    if (market== 1): ##cyb
        mr = '创业板'
    if (market== 2):  ##zz500
        mr = '中证500' 
    if (market== 3):   ##gz2000
        mr = '国证2000' 
    if (market== 4):   ##hs300
        mr = '沪深300' 
    if (market== 5):   ##zz1000
        mr = '中证1000' 
  
    title =mr+'-score result4:'
    rpspushfs(title)
    pushindustry(result4)
    rpspushfs(calcrpstxt(result4))
    title =mr+'-score result5:'
    rpspushfs(title)
    pushindustry(result5)
    rpspushfs(calcrpstxt(result5))
    title =mr+'-score result_max:'
    rpspushfs(title)
    pushindustry(result_max)
    rpspushfs(calcrpstxt(result_max))
    time2 = time.time() -time1        
    print(time2)
  # In[*]
    result_aaaa = df 
    result_aaaa.to_csv('1110.csv',index = False)
    print(df.dtypes)

    df = df.groupby('code', as_index=False).apply(obvtrend)
    df = df.reset_index(drop=True)
    
    df_obv_low =   df[(df['date'] ==today) & (df['OBV_trend_direction'] == "Up")&
                          ( ( df['op'] =='买入')|( df['op'] =='持仓') ) &
                          (df['rps120'] < 50) &( df['OBV_trend_length'] > 5)] 
    df_obv_tt =   df[(df['date'] ==today) & (df['OBV_trend_direction'] == "Up")&
                          ( df['OBV_trend_length'] > 5) &
                                                ( df['rps5'] > 85)]  
    
    result2s = df[(df['date'] ==today) & (df['industry2'] =='家电零部件Ⅱ') & (df['OBV_trend_length'] > 5) ]
    resultTTTTs = df[(df['date'] ==today) & (df['pchg'] >=15)  ]













   
    
    

    
    
