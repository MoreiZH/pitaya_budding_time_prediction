#!usr/bin/python3
# -*- coding: utf-8 -*- 
"""
Project: sa_pitaya
File: download_weather_daily.py
IDE: PyCharm
Creator: morei
Email: zhangmengleiba361@163.com
Create time: 2021-04-22 14:32
Introduction:
"""
import os
import gzip
import re
from datetime import datetime, timedelta
import csv
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def str2timestamp(s):
    try:
        return int(time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S")))
    except:
        return int(time.mktime(time.strptime(s, "%Y/%m/%d %H:%M:%S")))


def str2date(s):
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return dt

def timestamp2str(s):
    return time.strftime("%Y%m%d%H", time.localtime(int(s)))


def timestamp2str2(s):
    return time.strftime("%Y-%m", time.localtime(int(s)))


def create_hourly(df, year_start, year_end):
    dt = pd.date_range(start=f'{year_start}-1-1 00:00:00', end=f'{year_end}-12-31 23:00:00', freq="h")  # 构造每天小时时序
    date = [datetime.strftime(i, '%Y%m%d%H') for i in dt]  # 转换日期格式
    t_df = pd.DataFrame(zip(date), columns=['DATE_H'])
    m_df = t_df.merge(df, on='DATE_H', how='left')
    return m_df


def missing_value_completion(df, city, year_start, year_end):
    df['DATE_H'] = df.apply(lambda x: x['DATE'] + ' ' + str(x['HOUR']).zfill(2) + ':00:00', axis=1)
    df['DATE_H'] = df['DATE_H'].apply(lambda x: (str2date(x) + timedelta(hours=12)).strftime("%Y%m%d%H"))
    tt = create_hourly(df, year_start, year_end)
    for year in range(int(year_start)+1, int(year_end)+1):
        tt1 = tt[tt.DATE_H.str.startswith(str(year))]
        tt2 = tt1.interpolate()
        feat = ['DATE_H', 'TEMP']
        tt2 = tt2[feat].dropna()
        out1 = f'./output/completed_weather/{city}{year}.csv'
        if not Path(out1).parent.exists():
            os.makedirs(Path(out1).parent)
        tt2.to_csv(out1, index=False, encoding='utf-8')


def deal_csv_in_folder(folder_path):
    for p, ds, fs in os.walk(folder_path):
        item_l = [i.split(".")[0][:-4] for i in fs]
        year_l = [i.split(".")[0][-4:] for i in fs]
        df = pd.DataFrame(zip(item_l,year_l), columns=['city', 'year'])
        city_l = list(set(item_l))
        for c in city_l:
            temp_df = pd.DataFrame()
            tmp = df[df['city']==c]
            y_s = tmp['year'].min()
            y_e = tmp['year'].max()
            y_l = list(tmp['year'])
            for y in y_l:
                p_f = os.path.join(p, f'{c}{y}.csv')
                # 记录作物名称
                df1 = pd.read_csv(p_f)
                temp_df = pd.concat([temp_df,df1])
            missing_value_completion(temp_df, c, y_s, y_e)


if __name__ == "__main__":
    # input
    folder_path = './data/weather'
    # data completion
    deal_csv_in_folder(folder_path)



    a = 1
