# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:51:03 2023

@author: RYU
"""
import numpy as np
import pandas as pd


#%%
def mov_avg_filter(buffer, x_meas):
    
    """
    버퍼의 값과 현재 들어온 값을 받아 새로운 버퍼와 이동평균을 반출.
    
    x_meas: 현재 들어온 값.
    """
    n = len(buffer)
    for i in range(n-1):
        buffer[i] = buffer[i+1] # 현재 버퍼에 있는 값들을 한칸씩 앞으로 밀어 넣은 후
        buffer[n-1] = x_meas # 현재 들어온 값을 버퍼의 마지막에 배치
    x_avg = np.mean(buffer)
    return x_avg, buffer


#%%
def calculate_maf(n,signal):
    """
    n: 버퍼의 크기
    """
    result     = []
    sample_num = len(signal)
    
    for i in range(0,sample_num,1):
        x_meas = signal[i] # 현재의 singal 값
        
        if i == 0: #현재가 신호의 초기값이라면, 이동평균을 취할 수 없으므로
            x_avg, buffer = x_meas, x_meas * np.ones(n) # 초기 평균값과  버퍼는 초기값으로 채움.
        else:
            x_avg, buffer = mov_avg_filter(buffer, x_meas)
            
        #result[i] = x_avg
        result.append(x_avg)
        
    return np.array(result, dtype = 'float32')

#%%
def get_scaler(df, scaler_type, not_col, Target):

    scaler = scaler_type
    df_    = df.copy()
    
    col    =  [col for col in list(df_.columns) if col not in not_col]
    for x in col:
        tempt = df_[x]
        tempt = np.array(tempt)
        tempt = np.expand_dims(tempt, axis=-1)
        tempt = scaler.fit_transform(tempt)
        tempt = np.squeeze(tempt,axis=-1)
        
        """moving average"""
        if x == Target:
            tempt = calculate_maf(10,tempt)
            #tempt = tempt

        df_[x] = tempt
    
    return df_

#%%
from sklearn.preprocessing import RobustScaler
#from preprocessing import calculate_maf

def preprocessing(df, scaler_type):
    
    """filtering"""
    #df['Glucose_level'] = calculate_maf(12, df['Glucose_level'].values)
    #df['Glucose_level_original'] = calculate_maf(10, df['Glucose_level_original'].values)
    
    """normalization"""
    scaler = scaler_type
    df['Glucose_level'] = scaler.fit_transform(df[['Glucose_level']])
    
    
    """split ratio: 60%:20%:20%"""
    num_df = len(df)
    train_split = int(np.floor(0.6 * num_df))
    valid_split = int(np.floor((0.6+0.2) * num_df))
    
    train_df = df.iloc[:train_split].reset_index(drop=True)
    valid_df = df.iloc[train_split:valid_split].reset_index(drop=True)
    test_df  = df.iloc[valid_split:].reset_index(drop=True)
    
    return train_df, valid_df, test_df

#%%
def get_x_y_pairs(df, window_size, forcast_size, dawn=False):
    
    num = len(df) - window_size - forcast_size+1
    indexs = []
    count_over_300 = 0
    count_under_40 = 0
    count_nan = 0
    
    for idx in range(0, num, 1):
        
        target = df.iloc[idx+window_size:idx+window_size+forcast_size]
        
        """새벽 시간대만 선택"""
        if dawn == True:
            if target["Minutes"].values[0] >= 1*60 and target["Minutes"].values[0] <= 4*60 and target["Minutes"].values[-1] >= 4*60 and target["Minutes"].values[-1] <= 5*60:
                
                indexs.append(idx)
            else:
                continue

        total = df.iloc[idx:idx+window_size+forcast_size, :] # 나누기전 전체 구간
        
        """outlier """
        if any(i >= 300 for i in total['Glucose_level_original'].values):
            count_over_300 += 1
            continue
            
        elif any(i <= 40 for i in total['Glucose_level_original'].values):
            count_under_40 += 1
            continue
            
        elif 1 in list(total['Nan_point']):
            count_nan += 1
            continue
        else:
            indexs.append(idx)
             
        
    
    #print("300 이상이 있는 경우:", count_over_300)
    #print("40 이하가 있는 경우:", count_under_40)
    #print("Nan_point가 있는 경우:", count_nan)
    
    return indexs # 슬라이싱 시작 점.

#%%
"""
def adjust_independent_scaler(data_paths, scaler_type, not_col, Target, Filtering=False):
    
    result      = []
    scaler_dict = {}
    
    for i in range(0,len(data_paths),1):
        
        #init
        globals()['scaler_{}'.format(i)] = scaler_type
        
        #df = pd.read_csv(data_paths[i], index_col='Time')
        df = pd.read_csv(data_paths[i])
        
        
        df_    = df.copy()
        col    =  [col for col in list(df_.columns) if col not in not_col]
        
        
        #scaler & filtering
        for x in col:
            tempt = df_[x]
            tempt = np.array(tempt)
            tempt = np.expand_dims(tempt, axis=-1)
            tempt = globals()['scaler_{}'.format(i)].fit_transform(tempt)
            tempt = np.squeeze(tempt,axis=-1)

            if x == Target:
                if Filtering == True:
                    #trend     = calculate_maf(8,tempt) # 2시간 단위로 필터
                    #remainder = tempt - trend
                    #df_['trend']     = trend
                    #df_['remainder'] = remainder
                    
                    tempt = calculate_maf(8,tempt) # 2시간 단위로 필터
                else:
                    tempt = tempt

            df_[x] = tempt
        
        df_['scaler_num'] = i
        result.append(df_)
        scaler_dict[i] = globals()['scaler_{}'.format(i)]
        
        
    return result, scaler_dict
"""
#%%
"""
def adjust_independent_scaler_task2(data_paths, scaler_type, not_col, Target, Filtering=False):
    
    result      = []
    scaler_dict = {}
    
    for i in range(0,len(data_paths),1):
        
        #init
        globals()['scaler_{}'.format(i)] = scaler_type
        
        df = pd.read_csv(data_paths[i], index_col='Time')
        #df = pd.read_csv(data_paths[i])
        
        
        df_    = df.copy()
        col    =  [col for col in list(df_.columns) if col not in not_col]
        
        
        #scaler & filtering
        for x in col:
            tempt = df_[x]
            tempt = np.array(tempt)
            tempt = np.expand_dims(tempt, axis=-1)
            tempt = globals()['scaler_{}'.format(i)].fit_transform(tempt)
            tempt = np.squeeze(tempt,axis=-1)

            if x == Target:
                if Filtering == True:
                    tempt = calculate_maf(4,tempt) # 3시간 단위로 필터
                else:
                    tempt = tempt

            df_[x] = tempt
        
        df_['scaler_num'] = i
        result.append(df_)
        scaler_dict[i] = globals()['scaler_{}'.format(i)]
        
        
    return result, scaler_dict
"""
#%%
"""
def get_x_y_pairs(df, window_size, forcast_size, target):
    
 
    #train_scaled - training sequence data (df)
    #train_periods - How many data points to use as inputs
    #prediction_periods - How many periods to ouput as predictions
    
    #index_list: 시간 간격이 긴 구간을 체크한 nan point가 있는 행들의 인덱스.

    
    index_list = df.loc[df['Nan_point'] == 1].index
    x_train = []
    y_train = []
    
    
    for i in range(0, len(index_list)-1, 1):
        
        tempt1 = index_list[i]
        tempt2 = index_list[i+1]
        
        #인덱스 예외처리
        if i == 0:
            df_ = df.loc[:tempt1].copy().drop(['Nan_point'],axis=1)
        else:
            df_ = df.loc[tempt1+1 : tempt2].copy().drop(['Nan_point'],axis=1)
            
        #인덱스 예외처리2
        if len(df_) < window_size+forcast_size:
            continue
        else:
            pass
        
        #슬라이싱 구간
        #target_index = list(df_.columns).index(target)
    
        for idx in range(0, df_.shape[0]-window_size-forcast_size+1):

            x = df_.iloc[idx:idx+window_size, :].values

            #y = df_.iloc[idx+window_size:idx+window_size+forcast_size, target_index:target_index+1].values
            y = df_.iloc[idx+window_size:idx+window_size+forcast_size, :].values # teaching force 라서 다 있어야 됨.

            x_train.append(x)
            y_train.append(y)
    
    #return np.array(x_train, dtype='float32'), np.array(y_train, dtype='float32')
    return x_train, y_train
"""
#%%
"""
def get_x_y_pairs(df, window_size, forcast_size, target):
    
    
    train_scaled - training sequence data (df)
    train_periods - How many data points to use as inputs
    prediction_periods - How many periods to ouput as predictions
    
    

    x_train = []
    y_train = []

    for idx in range(0, df.shape[0]-window_size-forcast_size+1):
        
        
        #outlier pass
        total = df.iloc[idx:idx+window_size+forcast_size, :] # 나누기전 전체 구간
        if any(i >= 300 for i in total['Glucose_level_original'].values):
            continue
        
        
        
        x = df.iloc[idx:idx+window_size, :]
        if 1 in list(x['Nan_point']):
            continue
        x = x.values
        
        
        y = df.iloc[idx+window_size:idx+window_size+forcast_size, :] # teaching force 라서 다 있어야 됨.
        if 1 in list(y['Nan_point']):
            continue
        y = y.values
        


        

        x_train.append(x)
        y_train.append(y)
        
    return x_train, y_train
"""
#%%

"""
auto former, informer 등에 넣을 때는
입력으로 쓰인 window size의 반절을 디코더에서 참고하여 모델 결과를 내기 때문에 슬라이싱 구간을 일부 조정.

"""

"""

def get_x_y_pairs_for_transformers(df, window_size, forcast_size, target):
    
    
    train_scaled - training sequence data (df)
    train_periods - How many data points to use as inputs
    prediction_periods - How many periods to ouput as predictions
    
    

    x_train = []
    y_train = []

    for idx in range(0, df.shape[0]-window_size-forcast_size+1):
        
        
        #outlier pass
        total = df.iloc[idx:idx+window_size+forcast_size, :] # 나누기전 전체 구간
        if any(i >= 300 for i in total['Glucose_level_original'].values):
            continue
        
        x = df.iloc[idx:idx+window_size, :]
        if 1 in list(x['Nan_point']):
            continue
        x = x.values
        
        
        y = df.iloc[idx+int(window_size/2):idx+window_size+forcast_size, :] 
        if 1 in list(y['Nan_point']):
            continue
        y = y.values
        

        x_train.append(x)
        y_train.append(y)
        
    return x_train, y_train

"""

#%%
"""
def get_inverse_trnasform(model_type, numpy_target, numpy_pred, dictionary, idx_target, idx_scaler):
    
    
    numpy: .detach().cpu().numpy() 를 통해 변환된 텐서
    
    numpy_target ==> N Sequence Features(..., ..., target, scaler num)
    numpy_pred   ==> N Sequence
    
    dictionary: 앞에 프로세싱 과정 중 각 환자의 스케일러를 담은 dict
    idx_target: inverse transform을 적용하려고 하는 열의 위치
    idx_scaler: 스케일러 번호를 저장한 열의 위치
    
    
    n = numpy_target.shape[0] # 배치
    
    for i in range(0,n,1):
        
        #init
        tempt_target = numpy_target[i] # Sequence Features
        tempt_pred   = numpy_pred[i] # Sequence
        scaler_num   = int(tempt_target[:,idx_scaler][0]) #하나의 값임.
        
        #inverse transform to target
        scaled_value = tempt_target[:,idx_target]
        inverse_value = dictionary[scaler_num].inverse_transform(scaled_value.reshape(-1, 1))
        numpy_target[i][:,idx_target] = inverse_value.flatten()
        
        #inverse transform to pred
        scaled_value = tempt_pred
        inverse_value = dictionary[scaler_num].inverse_transform(scaled_value.reshape(-1, 1))
        
        
        if model_type == 'Linear':
            numpy_pred[i] = inverse_value.flatten()
        elif model_type == 'Transformer':
            numpy_pred[i] = inverse_value
        else:
            print("It is not defined")
            

    return numpy_target[:,:,idx_target], numpy_pred
"""
#%%
"""
def get_inverse_trnasform_for_data(numpy_target, dictionary, idx_target, idx_scaler):
    
    
    numpy: .detach().cpu().numpy() 를 통해 변환된 텐서
    
    numpy_target ==> N Sequence Features(..., ..., target, scaler num)
    numpy_pred   ==> N Sequence
    
    dictionary: 앞에 프로세싱 과정 중 각 환자의 스케일러를 담은 dict
    idx_target: inverse transform을 적용하려고 하는 열의 위치
    idx_scaler: 스케일러 번호를 저장한 열의 위치
    
    
    
    n = numpy_target.shape[0] # 배치
    
    for i in range(0,n,1):
        
        #init
        tempt_target = numpy_target[i] # Sequence Features
        scaler_num   = int(tempt_target[:,idx_scaler][0]) #하나의 값임.
        
        #inverse transform to target
        scaled_value = tempt_target[:,idx_target]
        inverse_value = dictionary[scaler_num].inverse_transform(scaled_value.reshape(-1, 1))
        numpy_target[i][:,idx_target] = inverse_value.flatten()

    return numpy_target
"""
