#!usr/bin/python3
# -*- coding: utf-8 -*- 
"""
Project: sa_pitaya_hourly
File: data_merger.py
IDE: PyCharm
Creator: morei
Email: zhangmengleiba361@163.com
Create time: 2021-05-08 16:16
Introduction:
this script is to create a hourly sunshine data, and match them with the temperature
"""
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def str2timestamp(s):
    try:
        return int(time.mktime(time.strptime(s, "%H:%M:%S")))
    except:
        return int(time.mktime(time.strptime(s, "%Y%m%d%H")))


def str2timestamp2(s):
    try:
        return int(time.mktime(time.strptime(s, "%Y-%m-%d")))
    except:
        return int(time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S")))


def timestamp2str(s):
    return time.strftime("%Y%m%d%H", time.localtime(int(s)))


def timestamp2str2(s):
    return time.strftime("%Y%m%d", time.localtime(int(s)))


def timestamp2daystr(s):
    return time.strftime("%Y/%m/%d", time.localtime(int(s)))


def str2duration(s):
    """字符串转为时间长度(/s)"""
    l = s.split(":")
    assert len(l) == 3  # %H:%M:%S
    res, factor = 0, 1
    while l:
        res += int(l.pop()) * factor
        factor *= 60
    return res


def static_accumulate(df, column_name="up_20deg"):
    df_tmp = df.copy()
    temp_list = []
    j = 0
    for i in range(len(df_tmp)):
        t = df_tmp.loc[i, column_name]

        if t > 0:
            j += t
            temp_list.append(j)
        else:
            j = 0
            temp_list.append(t)
    return temp_list


def create_hourly_sun(df):
    sun_df = pd.DataFrame(columns=['DATE_H', 'sun_time'])
    for i in range(len(df)):
        # 构造每一天的小时日照数据
        date, start_h, end_h, day_light, ave_t = df['DATE'][i], str(df['日出'][i]) + ':00', \
                                                 str(df['日落'][i]) + ':00', df['日照时长'][i], df['平均温度'][i]
        dt = pd.date_range(start=date + ' 00:00:00', end=date + ' 23:00:00', freq="H")  # 构造当天的24小时时序
        d = [datetime.strftime(i, '%Y%m%d%H') for i in dt]  # 转换日期格式
        h = [str2timestamp(datetime.strftime(i, '%H:%M:%S')) for i in dt]  # 提取小时信息
        s = np.repeat(1, len(d)).tolist()  # 假定每小时的日照均为1，后面根据日出日落时间进行调整
        delt = h[1] - h[0]
        light_duration = str2duration(day_light)/str2duration('1:00:00')
        ld = np.repeat(light_duration, len(d)).tolist()
        day_light_up_12 = 0 if light_duration < 12 else 1
        ld_up_12 = np.repeat(day_light_up_12, len(d)).tolist()
        day_ave = np.repeat(ave_t, len(d)).tolist()
        day_ave_up_18 = 0 if ave_t < 18 else 1
        ave_up_18 = np.repeat(day_ave_up_18, len(d)).tolist()
        df1 = pd.DataFrame(zip(d, h, s, ld, ld_up_12,day_ave,ave_up_18),
                           columns=['DATE_H', 'HOURstamp', 'sun_time', 'light_duration',
                                    'day_light_up_12', 'ave_t', 'ave_up_18'])
        df1.loc[(df1.HOURstamp < str2timestamp(start_h)) | (df1.HOURstamp > str2timestamp(end_h)), 'sun_time'] = 0.000000001  # 早于日出，晚于日落的日照均为0
        df1.loc[((df1.HOURstamp < str2timestamp(start_h) + delt) & (df1.HOURstamp > str2timestamp(start_h))), 'sun_time'] \
            = int(start_h.split(':')[1]) / 60  # 日出那个小时对应的分钟数，转小时
        df1.loc[(df1.HOURstamp < str2timestamp(end_h) + delt) & (df1.HOURstamp > str2timestamp(end_h)), 'sun_time'] \
            = int(end_h.split(':')[1]) / 60  # 日落那个小时对应的分钟数，转小时
        # df_t = df1[['DATE_H', 'sun_time']]
        sun_df = pd.concat([sun_df, df1])
    return sun_df


def merge_data_in_folder(weather_path, sun_path):
    for p, ds, fs in os.walk(weather_path):
        for f in fs:
            p_f = os.path.join(p, f)
            # 记录作物名称
            item = f.split('.')[0]
            # load data
            df1 = pd.read_csv(p_f)    # temperature
            df2 = pd.read_csv(os.path.join(sun_path, f))   # sun
            # ave_tem
            df1['DATE_H'] = df1['DATE_H'].apply(str)
            df_t = df1
            df_t['DATE'] = df_t['DATE_H'].apply(lambda x: timestamp2daystr(str2timestamp(x)))
            df1_d = df_t.groupby(df1['DATE'])['TEMP'].mean().reset_index()
            df1_d = df1_d.rename(columns={'TEMP': '平均温度'})
            # hourly sun
            sun_temp_daily = df2.merge(df1_d, on='DATE', how='left')
            sun_hourly = create_hourly_sun(sun_temp_daily)
            # merge
            df = df1.merge(sun_hourly, on='DATE_H', how='left')
            # output
            out1 = f'./output/sun_temperature_merged/{f}'
            if not Path(out1).parent.exists():
                os.makedirs(Path(out1).parent)
            df.to_csv(out1, index=False, encoding='utf-8')


if __name__ == "__main__":
    # input
    weather_path = './output/completed_weather'
    sun_path = './data/sun/'
    # merge data
    merge_data_in_folder(weather_path, sun_path)

    a = 1
