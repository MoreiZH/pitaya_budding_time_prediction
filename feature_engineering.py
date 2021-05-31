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
import math
from datetime import datetime, timedelta
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


def str2timestamp3(s):
    try:
        return int(time.mktime(time.strptime(s, "%Y/%m/%d")))
    except:
        return int(time.mktime(time.strptime(s, "%Y/%m/%d %H:%M:%S")))


def timestamp2str(s):
    return time.strftime("%Y%m%d%H", time.localtime(int(s)))


def timestamp2str2(s):
    return time.strftime("%Y%m%d", time.localtime(int(s)))


def timestamp2daystr(s):
    return time.strftime("%Y/%m/%d", time.localtime(int(s)))


def str2datetime(s):
    try:
        return datetime.strptime(s, "%Y/%m/%d")
    except:
        return datetime.strptime(s, "%Y%m%d%H")


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
    temp_list2 = []
    j = 0
    k = 0
    for i in range(len(df_tmp)):
        t = df_tmp.loc[i, column_name]
        t2 = df_tmp.loc[i, 'day_ave_up_25']

        if t > 0:
            j += t
            k += t2
            temp_list.append(j)
            temp_list2.append(k)
        else:
            j = 0
            k = 0
            temp_list.append(t)
            temp_list2.append(t2)
    return temp_list,temp_list2


### 日期前移
def date_forward_move(in_date, n_days = 25):
    dt = datetime.strptime(in_date, "%Y/%m/%d")
    out_date = (dt - timedelta(days=n_days)).strftime("%Y/%m/%d")
    return out_date


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
        day_ave = np.repeat(ave_t, len(d)).tolist()
        df1 = pd.DataFrame(zip(d, h, s, ld, day_ave),
                           columns=['DATE_H', 'HOURstamp', 'sun_time', 'light_duration','ave_t'])
        df1.loc[(df1.HOURstamp < str2timestamp(start_h)) |
                (df1.HOURstamp > str2timestamp(end_h)), 'sun_time'] = 0.000000000000000001  # 早于日出，晚于日落的日照均为0
        df1.loc[((df1.HOURstamp < str2timestamp(start_h) + delt) & (df1.HOURstamp > str2timestamp(start_h))), 'sun_time'] \
            = int(start_h.split(':')[1]) / 60  # 日出那个小时对应的分钟数，转小时
        df1.loc[(df1.HOURstamp < str2timestamp(end_h) + delt) & (df1.HOURstamp > str2timestamp(end_h)), 'sun_time'] \
            = int(end_h.split(':')[1]) / 60  # 日落那个小时对应的分钟数，转小时
        # df_t = df1[['DATE_H', 'sun_time']]
        sun_df = pd.concat([sun_df, df1])
    return sun_df


def create_hourly_add(b_df):
        add_d_s, add_d_e, add_h_s, add_h_e = b_df['light_date_start'], b_df['light_date_end'], \
                                             b_df['hour_start'] + ':00', b_df['hour_end'] + ':00'
        dt1 = pd.date_range(start=add_d_s + ' 00:00:00', end=add_d_e + ' 23:00:00', freq="H")  # 构造当天的24小时时序
        d = [datetime.strftime(i, '%Y%m%d%H') for i in dt1]  # 转换日期格式
        h = [str2timestamp(datetime.strftime(i, '%H:%M:%S')) for i in dt1]  # 提取小时信息
        s = np.repeat(1, len(d)).tolist()  # 假定每小时的日照均为1，后面根据日出日落时间进行调整
        delt = h[1] - h[0]
        add_duration = (str2duration(add_h_e) - str2duration(add_h_s)) / str2duration('1:00:00')
        a = np.repeat(add_duration, len(d)).tolist()
        df1 = pd.DataFrame(zip(d, h, s, a),
                           columns=['DATE_H', 'HOURstamp', 'add_time', 'add_duration'])
        df1.loc[(df1.HOURstamp < str2timestamp(add_h_s)) | (
                df1.HOURstamp > str2timestamp(add_h_e)), 'add_time'] = 0  # 早于日出，晚于日落的日照均为nan
        df1.loc[
            ((df1.HOURstamp < str2timestamp(add_h_s) + delt) & (df1.HOURstamp > str2timestamp(add_h_s))), 'add_time'] \
            = int(add_h_s.split(':')[1]) / 60  # 日出那个小时对应的分钟数，转小时
        df1.loc[(df1.HOURstamp < str2timestamp(add_h_e) + delt) & (df1.HOURstamp > str2timestamp(add_h_e)), 'add_time'] \
            = int(add_h_e.split(':')[1]) / 60
        df1 = df1[['DATE_H','add_time', 'add_duration']] # 只保留补光时间
        df1['add_duration'] = df1['add_duration'].apply(lambda x: 0.9*x)
        return df1


def feature_engineer(df):
    df['day_ave_up_18'] = df['ave_t'].apply(lambda x: 0 if x < 17 else 1)
    df['day_ave_up_25'] = df['ave_t'].apply(lambda x: 0 if x < 25 else 1)
    df['tem_up_18'] = df['TEMP'].apply(lambda x: 0 if x < 10 else 1)
    df['acc_tem_up_18'] = static_accumulate(df, 'tem_up_18')[0]
    df_d = df[['DATE','day_ave_up_18', 'day_ave_up_25']].drop_duplicates().reset_index(drop=True)
    sss = static_accumulate(df_d, 'day_ave_up_18')
    df_d['acc_day_ave_up_18'], df_d['sum_day_ave_up_25'] = sss[0], sss[1]
    df = df.merge(df_d, on=['DATE','day_ave_up_18', 'day_ave_up_25'], how='left')
    df.loc[(df.sum_day_ave_up_25 > 2), 'sum_day_ave_up_25'] = 5
    df_l = df[['DATE', 'total_light_up_12', 'day_ave_up_25']].drop_duplicates().reset_index(drop=True)
    df_l['acc_total_light_up_12'] = static_accumulate(df_l, 'total_light_up_12')[0]
    df = df.merge(df_l, on=['DATE', 'total_light_up_12', 'day_ave_up_25'], how='left')
    df.loc[(df.acc_day_ave_up_18 > 50), 'acc_day_ave_up_18'] = 50
    df.loc[(df.acc_total_light_up_12 > 50), 'acc_total_light_up_12'] = 50
    # df['tem_up_18*sun_time'] = (df['TEMP'] - 9)*df['light_duration'] * df['acc_day_ave_up_18'] * \
    #                            df['sun_time'] * df['day_ave_up_18'] * df['total_light_up_12']
    # df['tem_up_18*sun_time'] = (df['TEMP'] - 10) * (df['light_duration'] - 12) * df['acc_day_ave_up_18'] * \
    #                            df['sun_time'] * df['day_ave_up_18'] * df['total_light_up_12'] * df[
    #                                'acc_total_light_up_12']
    df['acc_total_light_up_12'] = df['acc_total_light_up_12'].apply(lambda x: math.log10(x + 1))
    df['tem_up_18*sun_time'] = (df['TEMP']-10) * (df['light_duration']-12) *\
                               df['sun_time']*df['day_ave_up_18']*df['total_light_up_12']*df['tem_up_18']
    df['acc_tem_up_18*sun_time'] = static_accumulate(df, 'tem_up_18*sun_time')[0]
    df['acc_tem_up_18'] = [math.log(i+1) for i in df['acc_tem_up_18']]
    df['feature'] = df['acc_tem_up_18*sun_time'] * df['acc_tem_up_18'] * df['acc_day_ave_up_18']/10000
    df['log(feature)'] = df['feature'].apply(lambda x: math.log10(x+1))
    feat = ['group', 'DATE_H', 'DATE', 'TEMP', 'ave_t', 'sun_time', 'light_duration', 'day_ave_up_18', 'day_ave_up_25',
            'total_light_up_12', 'tem_up_18', 'sum_day_ave_up_25', 'acc_tem_up_18', 'acc_total_light_up_12', 'acc_day_ave_up_18',
            'tem_up_18*sun_time', 'acc_tem_up_18*sun_time', 'feature', 'log(feature)']
    return df[feat]


def cal_loss(b_feature_df,feature_path):
    b_df = b_feature_df.copy()
    b_df['DATE'] = b_df['DATE'].apply(lambda x: str2datetime(str(x)))
    b_df['DATE_H'] = b_df['DATE_H'].apply(lambda x: str2datetime(str(x)))
    f = list(b_df['feature'])
    f_ave = np.mean(f)
    f_median = np.median(f)
    b_loss = []
    for p, ds, fs in os.walk(feature_path):
        for f in fs:
            p_f = os.path.join(p, f)
            # 记录作物名称
            item = f.split('.')[0]
            # load data
            df = pd.read_csv(p_f)    # feature
            df['DATE'] = df['DATE'].apply(lambda x: str2datetime(str(x)))
            df['DATE_H'] = df['DATE_H'].apply(lambda x: str2datetime(str(x)))
            if item == '南宁2018补光0':
                df = df[df['DATE']>'2018/11/1']
            df['f_d'] = df['feature'].apply(lambda x: abs(x-f_median))
            m = np.min(df['f_d'])
            i = 0
            pre_d = df[df['f_d'] == m]['DATE'].tolist()[i]
            b_d = b_df[b_df['group'] == item]['DATE'].tolist()[i]
            pre_h = df[df['f_d'] == m]['DATE_H'].tolist()[i]
            b_h = b_df[b_df['group'] == item]['DATE_H'].tolist()[i]
            d_loss = (pre_d - b_d).days
            h_loss = pre_h - b_h
            b_loss.append([item, b_d, b_h, pre_d, pre_h, d_loss, h_loss])
    df = pd.DataFrame(b_loss, columns=['group', 'DATE', 'DATE_H', 'PRE_DATE', 'PRE_DATE_H', 'DATE_LOSS', 'DATE_H_LOSS'])
    df.to_csv('./output/feature_engineered/batch_feature_loss.csv', index=False, encoding='utf-8')


def feature_engineer_in_folder(weather_path, sun_path, batch_df):
    batch_df['group'] = batch_df.apply(lambda x: x['city']+str(x['year']), axis=1)
    b_feature = pd.DataFrame()
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
            df1_d = df_t.groupby(df_t['DATE'])['TEMP'].mean().reset_index()
            df1_d = df1_d.rename(columns={'TEMP': '平均温度'})
            # hourly sun
            df2['DATE'] = df2['DATE'].apply(lambda x: timestamp2daystr(str2timestamp3(x)))
            sun_temp_daily = df2.merge(df1_d, on='DATE', how='left')
            sun_hourly = create_hourly_sun(sun_temp_daily)
            batch = batch_df[batch_df['group']==item]
            # merge
            for i in range(len(batch)):
                b_df = batch.iloc[i]
                if b_df['add_light']:
                    add_hourly = create_hourly_add(b_df)
                    tmp = sun_hourly.merge(add_hourly, on='DATE_H', how='left')
                    tmp.loc[(tmp.add_time > 0), 'sun_time'] = tmp.loc[(tmp.add_time > 0), 'add_time'] + \
                                                              tmp.loc[(tmp.add_time > 0), 'sun_time']  # 早于日出，晚于日落的日照均为nan
                    tmp.loc[(tmp.sun_time > 1), 'sun_time'] = 1
                    tmp.loc[(tmp.add_duration > 0), 'light_duration'] = tmp.loc[(tmp.add_duration > 0), 'light_duration'] + \
                                                                        tmp.loc[(tmp.add_duration > 0), 'add_duration']
                    sun_df = tmp
                    sun_df['group'] = f'{item}补光{i}'
                    out1 = f'./output/feature_engineered/cities/{item}补光{i}.csv'
                else:
                    sun_df = sun_hourly
                    sun_df['group'] = f'{item}'
                    out1 = f'./output/feature_engineered/cities/{item}.csv'
                # feature engineering
                df = df1.merge(sun_df, on='DATE_H', how='left')
                df = feature_engineer(df)
                b_time = timestamp2str(str2timestamp3(b_df['batch_time'] + ' 10:00:00'))
                b = df[df['DATE_H'] == b_time]
                b_feature = pd.concat([b_feature,b])
                # output
                if not Path(out1).parent.exists():
                    os.makedirs(Path(out1).parent)
                df.to_csv(out1, index=False, encoding='utf-8')
    b_feature.to_csv('./output/feature_engineered/batch_feature.csv', index=False, encoding='utf-8')
    cal_loss(b_feature, Path(out1).parent)


if __name__ == "__main__":
    # input
    weather_path = './output/completed_weather'
    sun_path = './data/sun/'
    batch_df = pd.read_csv('./data/BatchTime_addlight/BatchTime_addlight.csv')
    # b_df = batch_df.loc[4]
    # merge data
    feature_engineer_in_folder(weather_path, sun_path, batch_df)


    a = 1
