import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from datetime import datetime, timedelta
import pickle
"""suggestion：Set all data file names uniformly as "device0x_100x_actName”"""

"""_summary_Rename the sensor raw data files directly on the original files，no return
    Dataset File Rules：Categorizing them by activity type, Naming folders according to the rule "100x_ActivityName", include devices："device0x_100x_jump"
    parameter-for example：
    ①device_map_dict = {
    'WT901BLE67(de.aa.8a.1c.bb.6c)': 'device01',
    'WT901BLE67(f3.12.3b.05.4e.75)': 'device02',
    'WT901BLE67(c8.5a.a5.e1.39.3b)': 'device03'}
    ②folder_path = r"D:\projects\Har-datasets\thidssets-nine\original"
"""
def renameFilename(folder_path, device_map_dict):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                file_name, file_extension = os.path.splitext(file)
                devicename = file_name.split("_")[0]
                new_devicename = device_map_dict.get(devicename, devicename)
                last_folder_name = os.path.basename(root)
                new_filename = f"{new_devicename}_{last_folder_name}{file_extension}"
                new_file_path = os.path.join(root, new_filename)
                print(f"path of the source file：{file_path}")
                print(f"path of the renamed file：{new_file_path}")
                os.rename(file_path, new_file_path)

"""_summary_Name the additional classification marker column after row-wise fusion (named based on the missing position)
    return classify_tag: the column name
"""
def mergeRow_classify_tag(missing_pos):
    if missing_pos == 0:
        classify_tag = 'deviceid'
    elif missing_pos == 1:
        classify_tag = 'subjectid'
    elif missing_pos == 2:
        classify_tag = 'activityid'
    else:
        classify_tag = 'None'
    return classify_tag

"""_summary_extracting the corresponding names as markers when fusing data, and putting them into a new file(such as: classify_files_mergeRow)
    return new_file_name, missing_parts_dict
    """
def extract_missing_parts(original_file_name, current_name):
    original_parts = original_file_name.split('_')
    print(original_parts)
    current_parts = current_name.split('_')
    missing_parts_dict = {}
    pos_flag = -1
    for part in original_parts:
        pos_flag = pos_flag + 1
        if part not in current_parts:
            classify_tag = mergeRow_classify_tag(pos_flag)
            missing_parts_dict[classify_tag] = part
            current_parts.insert(pos_flag,f'merge{classify_tag}')#such as: 'device01_1001_mergeactivityid'
    new_file_name = '_'.join(current_parts)# File name after fusion: Use specified values to replace the missing parts of the name
    #return: the file name after fusion, specific missing values (the values of the column after fusion need to be set)
    return new_file_name, missing_parts_dict

"""_summary_Custom Rule Function: “device0x_100x_actNmae”
    files_by_rule = classify_files_by_rule_1(folder_path, custom_rule)
"""
"""_summary_Custom Rule Function: classified by subject & activity
    return：rules for file names
"""
def rule_func_bySubAct(file_name):
    parts = file_name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:3])
    else:
        return "other"  
    
"""_summary_Custom Rule Function: classified by device
"""
def custom_rule_bydevice(file_name):
    parts = file_name.split("_")
    if len(parts) >= 3:
        return parts[0]
    else:
        return "other"
    
"""_summary_Custom Rule Function: classified by device & activity
"""
def custom_rule_bydeviceAct(file_name):
    parts = file_name.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[2]}"
    else:
        return "other"

"""_summary_Custom Rule Function: classified by activity
"""
def custom_rule_bydeviceSub(file_name):
    parts = file_name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    else:
        return "other"

"""_summary_Custom Rule Function: Fuse the data from the three devices for the same activity, such as--"device0123_actName"
"""
def custom_rule_byAct_mergesub(file_name):
    parts = file_name.split("_")
    if len(parts) >= 3:
        return f"{parts[1]}"
    else:
        return "other"

"""_summary_Classification of conventional naming methods
"""
def custom_rule_byAct(file_name):
    parts = file_name.split("_")
    if len(parts) >= 3:
        return f"{parts[2]}"
    else:
        return "other"

"""_summary_Operations on the KU-HAR dataset
    Merge all data files for the same activity, "100x_actID_other"
"""
def custom_rule_byKuAct(file_name):
    parts = file_name.split("_")
    if len(parts) >= 3:
        return f"{parts[1]}"
    else:
        return "other"
        
"""_summary_Analyze each incoming file individually and classify them one by one according to the rules
    ①folder_path：The folder where all the files to be classified are located
    ①rule_func
    return files_by_rule: List of files (full paths) classified according to the rules
"""
def classify_files_by_rule_func(folder_path, rule_func):
    files_by_rule = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            rule = rule_func(file_name)
            if rule not in files_by_rule:
                files_by_rule[rule] = []
            files_by_rule[rule].append(file_path)
    return files_by_rule

"""_summary_Only extract files under the specified category
    Add a new check: determine if the current file name complies with the specified rules. If it does, include the file; otherwise, exclude it.
    return files_by_rule: Files containing specified categories
"""
def classify_files_by_rule_func_find(folder_path, rule_func, find_rule):
    files_by_rule = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            rule = rule_func(file_name)
            if rule == find_rule:
                if rule not in files_by_rule:
                    files_by_rule[rule] = []
                files_by_rule[rule].append(file_path)
    return files_by_rule

 
"""_summary_Traverse the rule list file and output the corresponding key-values
    ①files_by_rule
"""
def print_files(files_by_rule):
    for key, file_list in files_by_rule.items():
        print(f"rule {key}：")
        for file_path in file_list:
            print(file_path)

"""_summary_The number of samples that can be extracted according to the specified step size (split_num)
param——folder_path: Count the total number of data lines, the number of files, time period, the number of lines and samples of each file in a folder.
"""
def calcrowfolder(folder_path, split_time):
    file_count = 0
    num_rows = 0
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                df = pd.read_csv(file_path, header=0,index_col=False,encoding='utf-8')
                num_rows = int(df.shape[0]/split_time) + num_rows
                start_row_value = df.iloc[0, 0].strip()
                last_row_value = df.iloc[-1, 0].strip()
                row_count = df.shape[0]
                time = row_count / split_time
                timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')
                print("path of file：",file_path)
                print(f"rows and sample:{row_count}, {int(row_count/split_time)} \n timediff：{timediff} \n time period:{start_row_value}-{last_row_value}")
    print('the total number of files in this folder：',file_count,'（Original number of samples）')
    print('The total number of lines of all files in this folder：',num_rows,'（the extracted subsample）')

"""_summary_Count the data of the specified category
"""
def classify_files_calcrowfolder_byrule(folder_path,split_time,rule_func,find_rule):
    files_by_rule = classify_files_by_rule_func_find(folder_path, rule_func,find_rule)
    for key, file_list in files_by_rule.items():
        print(f"******rule {key}******")
        for file_path in file_list:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=0,index_col=False,encoding='utf-8')
                start_row_value = df.iloc[0, 0].strip()
                last_row_value = df.iloc[-1, 0].strip()
                row_count = df.shape[0]
                time = row_count/split_time
                timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')
                print("path of file：",file_path)
                print(f"rows & samples: {row_count},{int(row_count/split_time)} \n timediff: {timediff} \n time period:{start_row_value}-{last_row_value}")

"""_summary_the information of different devices for the same subject and activity
"""
def classify_files_calcrowfolder(input_path,split_time,rule_func):
    if os.path.isdir(input_path):
        print(f"**********path of file: {input_path}**********")
        files_by_rule = classify_files_by_rule_func(input_path, rule_func)
        for key, file_list in files_by_rule.items():
            print(f"******rule {key}******")
            for file_path in file_list:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, header=0,index_col=False,encoding='utf-8')
                    start_row_value = df.iloc[0, 0].strip()
                    last_row_value = df.iloc[-1, 0].strip()
                    row_count = df.shape[0]
                    time = row_count/split_time
                    timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')
                    print("path of file：",file_path)
                    print(f"rows & samples:{row_count},{int(row_count/split_time)} \n timediff：{timediff} \n time period:{start_row_value}-{last_row_value}")
    elif os.path.isfile(input_path):
        print(f"**********path of file:{input_path}**********")
        df = pd.read_csv(input_path, header=0,index_col=False,encoding='utf-8')
        start_row_value = df.iloc[0, 0].strip()
        last_row_value = df.iloc[-1, 0].strip()
        row_count = df.shape[0]
        time = row_count/split_time
        timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')  
        print("path of file：",input_path)
        print(f"rows & samples:{row_count},{int(row_count/split_time)} \n timediff：{timediff} \n time period:{start_row_value}-{last_row_value}")
    else:
        print(f"{input_path} not a valid file or folder path")

"""_summary_Uniform cutting
"""
def cutfolderrow(folder_path, startcutrow, endcutrow,split_time):
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root,file)
                df = pd.read_csv(file_path, header = 0, index_col = False, encoding = 'utf-8')
                df = df.iloc[startcutrow:df.shape[0]-endcutrow,:]
                start_row_value = df.iloc[0, 0].strip()
                last_row_value = df.iloc[-1, 0].strip()
                row_count = df.shape[0]
                time = row_count/split_time
                timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')
                print("The cropped file path：",file_path)
                print(f"The number of trimmed rows & samples:{row_count},{int(row_count/split_time)} \n timediff：{timediff} \n time period:{start_row_value}-{last_row_value}")
                df.to_csv(file_path,header = True, index = False)


def cal_rowcut(folder_path, startcutrow, endcutrow,split_time):
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root,file)
                df = pd.read_csv(file_path, header = 0, index_col = False, encoding = 'utf-8')
                df = df.iloc[startcutrow:df.shape[0]-endcutrow,:]
                start_row_value = df.iloc[0, 0].strip()
                last_row_value = df.iloc[-1, 0].strip()
                row_count = df.shape[0]
                time = row_count/split_time
                timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')
                print("The cropped file path：",file_path)
                print(f"The number of trimmed rows & samples:{row_count},{int(row_count/split_time)} \n timediff：{timediff} \n time period:{start_row_value}-{last_row_value}")

"""_summary_Delete the beginning and end data of all files that comply with a certain specified rule
    folder_path
    startcutrow:the number of rows of the first data
    endcutrow: the number of rows of the tail data
    rule_func
    find_rule: A single category that needs to be operated on
"""
def classify_files_cutfolderrow_byrule(folder_path, startcutrow, endcutrow, split_time,rule_func, find_rule):
    files_by_rule = classify_files_by_rule_func_find(folder_path, rule_func, find_rule)
    for key, file_list in files_by_rule.items():
            print(f"******rule {key}******")
            for file_path in file_list:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path, header = 0, index_col = False, encoding = 'utf-8')
                df = df.iloc[startcutrow:df.shape[0]-endcutrow,:]
                start_row_value = df.iloc[0, 0].strip()
                last_row_value = df.iloc[-1, 0].strip()
                row_count = df.shape[0]
                time = row_count/split_time
                timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')
                print("The cropped file path：",file_path)
                print(f"The number of trimmed rows & samples:{row_count},{int(row_count/split_time)} \n timediff：{timediff} \n time period:{start_row_value}-{last_row_value}")
                df.to_csv(file_path,header = True, index = False)
                print("————Clipping successfully————")

"""Classification of cutting
"""
def classify_files_cutfolderrow(input_path, startcutrow, endcutrow,split_time, rule_func):
    if os.path.isdir(input_path):
        print(f"**********path of file:{input_path}**********")
        files_by_rule = classify_files_by_rule_func(input_path, rule_func)
        for key, file_list in files_by_rule.items():
                print(f"******rule {key}******")
                for file_path in file_list:
                    if file_path.endswith(".csv"):
                        df = pd.read_csv(file_path, header = 0, index_col = False, encoding = 'utf-8')
                    df = df.iloc[startcutrow:df.shape[0]-endcutrow,:]
                    start_row_value = df.iloc[0, 0].strip()
                    last_row_value = df.iloc[-1, 0].strip()
                    row_count = df.shape[0]
                    time = row_count/split_time
                    timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')
                    print("The cropped file path：",file_path)
                    print(f"The number of trimmed rows & samples:{row_count},{int(row_count/split_time)} \n timediff：{timediff} \n time period:{start_row_value}-{last_row_value}")
                    df.to_csv(file_path,header = True, index = False)
                    print("————Clipping successfully————")
    elif os.path.isfile(input_path):
        print(f"**********path of folder:{input_path}**********")
        df = pd.read_csv(file_path, header = 0, index_col = False, encoding = 'utf-8')
        df = df.iloc[startcutrow:df.shape[0]-endcutrow,:]
        start_row_value = df.iloc[0, 0].strip()
        last_row_value = df.iloc[-1, 0].strip()
        row_count = df.shape[0]
        time = row_count/split_time
        timediff = datetime.strptime(last_row_value,'%H:%M:%S.%f') - datetime.strptime(start_row_value,'%H:%M:%S.%f')
        print("The cropped file path：",file_path)
        print(f"The number of trimmed rows & samples:{row_count},{int(row_count/split_time)} \n timediff：{timediff} \n time period:{start_row_value}-{last_row_value}")
        df.to_csv(file_path,header = True, index = False)
   
"""_summary_Replace the column names and delete the redundant columns
"""
def change_colname_path(folder_path, new_folder_path, x_head_key, del_head_key):
    Flag = True
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root,file)
                df = pd.read_csv(file_path,header = 0,index_col=False)
                if Flag:
                    print(f'The column names of the original file：\n{df.columns}')
                num_columns = df.shape[1]
                if num_columns != len(x_head_key):
                    df = df.iloc[:, 0:len(x_head_key)]
                df.columns = x_head_key
                df = df.drop(del_head_key, axis = 1)
                if Flag:
                    Flag = False
                new_root_folder = os.path.basename(root)
                new_folder = os.path.join(new_folder_path,new_root_folder)
                new_file_name = file
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_file_path = os.path.join(new_folder,new_file_name)
                try: 
                    f = open(new_file_path, 'w+') 
                    if f: 
                        f.truncate()
                    df.to_csv(new_file_path,header=True,index=False)
                except UnicodeEncodeError: 
                    print("Encoding error. The data cannot be written to the file, so it is directly ignored")

"""Replace the column names and delete the redundant columns
x_head_key = ['timeStamp', 'deviceName', 'deviceTime', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'angleX', 'angleY', 'angleZ', 'magnX', 'magnY', 'magnZ', 'temp', 'orien0', 'orien1', 'orien2', 'orien3']
return new_df: the modified df-type data
"""
def change_colname_df(df, x_head_key, del_head_key):
    num_columns = df.shape[1]
    if num_columns != len(x_head_key):
        df = df.iloc[:, 0:len(x_head_key)]
    df.columns = x_head_key
    df = df.drop(del_head_key, axis = 1)
    return df
    
"""_summary_Reordering of the time column
"""
def updateTimeStamps_path(folder_path, new_folder_path, timeStamp_column_name):
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root,file)
                file_path = os.path.join(root,file)
                df = pd.read_csv(file_path, header = 0, index_col = False, encoding = 'utf-8')
                df[timeStamp_column_name] = df[timeStamp_column_name].str.strip()
                timeStamp_start = pd.to_datetime(df[timeStamp_column_name].iloc[0], format = '%H:%M:%S.%f')
                time_increment = pd.timedelta_range(start='0', periods=len(df), freq='10ms')
                df[timeStamp_column_name] = timeStamp_start + time_increment
                df[timeStamp_column_name] = df[timeStamp_column_name].apply(lambda x: pd.to_datetime(x, format="%H:%M:%S.%f").strftime("%H:%M:%S.%f"))
                new_root_folder = os.path.basename(root)
                new_folder = os.path.join(new_folder_path,new_root_folder)
                new_file_name = file
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_file_path = os.path.join(new_folder,new_file_name)
                try: 
                    f = open(new_file_path, 'w+') 
                    if f: 
                        f.truncate()
                    df.to_csv(new_file_path,header=True,index=False)
                except UnicodeEncodeError: 
                    print("Encoding error. The data cannot be written to the file, so it is directly ignored")

"""_summary_the reordering of adding time columns
"""
def updateTimeStamps_df(df, timeStamp_column_name):
    df[timeStamp_column_name] = df[timeStamp_column_name].str.strip()
    timeStamp_start = pd.to_datetime(df[timeStamp_column_name].iloc[0], format = '%H:%M:%S.%f')
    time_increment = pd.timedelta_range(start='0', periods=len(df), freq='10ms')
    df[timeStamp_column_name] = timeStamp_start + time_increment
    df[timeStamp_column_name] = df[timeStamp_column_name].apply(lambda x: pd.to_datetime(x, format="%H:%M:%S.%f").strftime("%H:%M:%S.%f"))
    return df

"""_summary_Replace the data of a specified column
"""
def replace_coldata_path(folder_path, new_folder_path, column_name, mapping_dict):
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root,file)
                file_path = os.path.join(root,file)
                df = pd.read_csv(file_path, header = 0, index_col = False, encoding = 'utf-8')
                keys = list(mapping_dict.keys())
                values = list(mapping_dict.values())  
                for index, row in df.iterrows():
                    for column_name, value in row.iteritems():
                        if value == keys[0]:
                            df.loc[df[column_name].isin([value]),column_name]= values[0]
                        if value == keys[1]:
                            df.loc[df[column_name].isin([value]),column_name]= values[1]
                        if value == keys[2]:
                            df.loc[df[column_name].isin([value]),column_name]= values[2]
                new_root_folder = os.path.basename(root)
                new_folder = os.path.join(new_folder_path,new_root_folder)
                new_file_name = file
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_file_path = os.path.join(new_folder,new_file_name)
                try: 
                    f = open(new_file_path, 'w+') 
                    if f: 
                        f.truncate()
                    df.to_csv(new_file_path,header=True,index=False)
                except UnicodeEncodeError: 
                    print("Encoding error. The data cannot be written to the file, so it is directly ignored")

"""_summary_Replace the data of the column with the specified column name with the target column data.
"""
def replace_coldata_df(df, column_name, mapping_dict):
    keys = list(mapping_dict.keys())
    values = list(mapping_dict.values())  
    for index, row in df.iterrows():
        for column_name, value in row.iteritems():
            if value == keys[0]:
                df.loc[df[column_name].isin([value]),column_name]= values[0]
            if value == keys[1]:
                df.loc[df[column_name].isin([value]),column_name]= values[1]
            if value == keys[2]:
                df.loc[df[column_name].isin([value]),column_name]= values[2]
    return df 

"""_summary_Replace column names, delete redundant columns, add time column reordering, and replace data in the deviceName column.
    device_map_dict = {
    'WT901BLE67(de:aa:8a:1c:bb:6c)': 'device01',
    'WT901BLE67(f3:12:3b:05:4e:75)': 'device02',
    'WT901BLE67(c8:5a:a5:e1:39:3b)': 'device03'}
    x_head_key = ['timeStamp', 'deviceName', 'deviceTime', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'angleX', 'angleY', 'angleZ', 'magnX', 'magnY', 'magnZ', 'temp', 'orien0', 'orien1', 'orien2', 'orien3']  
    del_head_key = ['deviceTime'] 
"""
def colName_tStamp_dName(old_folder_path,new_folder_path,device_map_dict,x_head_key, del_head_key):
    for root,dirs,files in os.walk(old_folder_path):
        for file in files:
            if file.endswith(".csv"):
                old_file_path = os.path.join(root,file)
                df = pd.read_csv(old_file_path, header = 0, index_col = False, encoding = 'UTF-8')
                df = change_colname_df(df, x_head_key, del_head_key)
                
                timeStamp_column_name = x_head_key[0]
                df = updateTimeStamps_df(df, timeStamp_column_name)
                
                df = replace_coldata_df(df, x_head_key[1], device_map_dict)
                df[timeStamp_column_name] = df[timeStamp_column_name].astype(str) 
                
                new_file_name = file
                new_root_folder = os.path.basename(root)
                new_folder = os.path.join(new_folder_path, new_root_folder)
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_file_path = os.path.join(new_folder,new_file_name)
                try: 
                    f = open(new_file_path, 'w+') 
                    if f: 
                        f.truncate()
                    df.to_csv(new_file_path,header=True,index=False)
                except UnicodeEncodeError: 
                    print("Encoding error. The data cannot be written to the file, so it is directly ignored")

"""_summary_Retain specified columns through a list of column names.
:params old_folder_path
:params old_folder_path
:params keep_columns:keep_columns = ['timeStamp', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
return df
"""
def filter_column(old_folder_path, new_folder_path, keep_columns):
    for root, dirs, files in os.walk(old_folder_path):
        for file in files:
            if file.endswith(".csv"):
                old_file_path = os.path.join(root, file)
                df = pd.read_csv(old_file_path, header=0,index_col=False,encoding='utf-8')
                df = df.loc[:, keep_columns]
                df = df.drop('timeStamp',axis = 1)
                
                device_folder_name = os.path.basename(root)
                new_folder = os.path.join(new_folder_path, device_folder_name)
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_file_name = file
                new_file_path = os.path.join(new_folder,new_file_name)
                print(df)
                try: 
                    f = open(new_file_path, 'w+') 
                    if f: 
                        f.truncate()
                    df.to_csv(new_file_path,header=True,index=False)
                except UnicodeEncodeError: 
                    print("Encoding error. The data cannot be written to the file, so it is directly ignored")

"""_summary_ Reclassify files according to the passed rules and output new paths
    header:0/None
    rule_func = custom_rule_bydevice、custom_rule_byKuAct / rule_func_bySubAct、custom_rule_bydeviceAct ect..
"""
def classify_files_to_newpath(folder_path, new_folder_path, rule_func,header):
    files_by_rule = classify_files_by_rule_func(folder_path, rule_func)
    for key, file_list in files_by_rule.items():
        print(f"rule {key}：")
        for file_path in file_list:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=header,index_col=False,encoding='utf-8')
                file_name = os.path.basename(file_path)
                new_devicefolder_path = os.path.join(new_folder_path, key)
                if not os.path.exists(new_devicefolder_path):
                    os.makedirs(new_devicefolder_path)
                new_file_path = os.path.join(new_devicefolder_path,file_name)
                try: 
                    f = open(new_file_path, 'w+') 
                    if f: 
                        f.truncate()
                    if header == None:
                        df.to_csv(new_file_path,header = False, index = False)
                    elif header == 0:
                        df.to_csv(new_file_path,header=True,index=False)
                    else:
                        print(f'Failed to write: {new_file_path}')
                except UnicodeEncodeError: 
                    print("Encoding error. The data cannot be written to the file, so it is directly ignored")

"""_summary_merge relevant data according to custom rules, and splice and merge the classified data row by row
    rule_func = custom_rule_bydeviceAct、custom_rule_byKuAct(ku-Har) / rule_func_bySubAct、custom_rule_bydevice etc..
    header: 0/None
"""
def classify_files_mergeRow(folder_path, new_folder_path, rule_func,header):
    files_by_rule = classify_files_by_rule_func(folder_path, rule_func)
    for key, file_list in files_by_rule.items():
        print(f"rule {key}：") 
        dfs = []
        for file_path in file_list:
            flag = True
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=header,index_col=False,encoding='utf-8')
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                new_file_name, missing_parts_dict = extract_missing_parts(file_name, key)
                for mapkey in missing_parts_dict.keys():
                    mapvalue = missing_parts_dict[mapkey]
                    mapkey_data = np.repeat(mapvalue,df.shape[0]).reshape(-1, 1)
                    df[mapkey] = mapkey_data
                dfs.append(df)
                if flag:
                    new_subfolder_name = os.path.basename(os.path.dirname(file_path))
                    flag = False
                else:
                    pass
        new_subfolder_path = os.path.join(new_folder_path, new_subfolder_name)
        new_df = pd.concat(dfs, ignore_index=True)
        if not os.path.exists(new_subfolder_path):
            os.makedirs(new_subfolder_path)
        new_file_path = os.path.join(new_subfolder_path,f"{new_file_name}.csv")
        try: 
            f = open(new_file_path, 'w+') 
            if f: 
                f.truncate()
            if header == None:
                new_df.to_csv(new_file_path,header = False, index = False)
            elif header == 0:
                new_df.to_csv(new_file_path,header=True,index=False)
            else:
                print(f'Failed to write: {new_file_path}')
        except UnicodeEncodeError: 
            print("Encoding error. The data cannot be written to the file, so it is directly ignored")


"""
retain data in specified columns, group data of the same person from files of the same device, and identify each activity category with act_id
"""
def merge_subject_flagactid(folder_path, new_folder_path, keep_columns, csv_patterns,rule_func,namepart_num):
    
    files_by_rule = classify_files_by_rule_func(folder_path, rule_func)
    for key, file_list in files_by_rule.items():
        print(f"rule {key}：") 
        dfs = []
        for file_path in file_list:
            flag = True
            print("path of file: ",file_path)
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=0,index_col=False,encoding='utf-8')
                keep_columns_df = df[keep_columns]
                df_copy = keep_columns_df.copy() 
                original_file_name = os.path.splitext(os.path.basename(file_path))[0]
                subject_parts = original_file_name.split("_")
                if len(subject_parts) >= 3:
                    actid = csv_patterns[subject_parts[namepart_num]]
                else:
                    actid = 'null'
                actid_data = np.repeat(actid,df_copy.shape[0]).reshape(-1, 1)
                df_copy.loc[:, 'act_id'] = actid_data
                dfs.append(df_copy)
                if flag:
                    flag = False
                else:
                    pass
        new_df = pd.concat(dfs, ignore_index=True)
        
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        new_file_path = os.path.join(new_folder_path,f"subject{subject_parts[1]}.csv")
        try: 
            f = open(new_file_path, 'w+') 
            if f: 
                f.truncate()
                new_df.to_csv(new_file_path,header=True,index=False)
            else:
                print(f'Failed to write: {new_file_path}')
        except UnicodeEncodeError: 
            print("Encoding error. The data cannot be written to the file, so it is directly ignored")

"""_summary_Concatenate the classified data files column-wise (by column)
    rule_func = rule_func_bySubAct
    Target file name: Distinguish by device
"""
def classify_files_mergeCol(folder_path, new_folder_path, keep_columns,rule_func):
    files_by_rule = classify_files_by_rule_func(folder_path, rule_func)
    for key, file_list in files_by_rule.items():
        dfs = []
        for file_path in file_list:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=0,index_col=False,encoding='utf-8')
                keep_columns_df = df[keep_columns]

                original_file_name = os.path.splitext(os.path.basename(file_path))[0]
                new_file_name, missing_parts_dict = extract_missing_parts(original_file_name, key)
                for mapkey in missing_parts_dict.keys():
                    mapvalue = missing_parts_dict[mapkey]
                    new_columns = {col: mapvalue+'_' + col for col in keep_columns_df.columns}
                    keep_columns_df = keep_columns_df.rename(columns=new_columns)
                dfs.append(keep_columns_df)

        merged_df = pd.concat(dfs, axis=1, ignore_index=False)
        merged_df = merged_df.dropna()       
        new_file_path = os.path.join(new_folder_path,f"deviceMerge_{key}.csv")
        try: 
            f = open(new_file_path, 'w+') 
            if f: 
                f.truncate()
            merged_df.to_csv(new_file_path,header=True,index=False)
        except UnicodeEncodeError: 
            print("Encoding error. The data cannot be written to the file, so it is directly ignored")

"""Convert each window of data into a single row representation
:params csv_folder: Path of the folder where the file is located
:params windows_width: eg, windows_width = 300
:params step
return result_df: 1–300、301–600、601–900 ➞ acc X、Y、Z; 901–1200、1201–1500、1501–1800 ➞ gyro X、Y、Z
                1801 ➞ act ID(0-11); 1802 ➞ Length of data for each channel in the subsample (windows_width); 1803 ➞ Serial number of the subsample
"""
def transformcsv(csv_patterns, folder_path,windows_width,step,keep_columns):
    csv_files = {}
    for csv_file in os.listdir(folder_path):
        for pattern, category in csv_patterns.items():
            if csv_file.endswith(pattern):
                csv_files.setdefault(category, []).append(os.path.join(folder_path, csv_file))

    csv_data = {}
    for key in csv_files:
        csv_data[key] = []
        for csv_file in csv_files[key]:
            df = pd.read_csv(csv_file,header=0,index_col=False,encoding='utf-8')
            df = df[keep_columns]
            csv_data[key].append(df)
    result_df = pd.DataFrame()
    index = 1
    for key, dfs in csv_data.items():
        for df in dfs:
            windows = []
            for i in range(0, len(df), step):
                window = df.iloc[i:i+windows_width]
                if len(window) == windows_width:
                    windows.append(window)

            for df in windows:
                concat_df = pd.concat([df[col] for col in df], ignore_index=True).to_frame().T
                category_id = key
                channel_lengths = windows_width
                metadata_df = pd.DataFrame([[category_id, channel_lengths, index]])
                row_data = pd.concat([concat_df, metadata_df], axis=1)
                row_data.columns = range(row_data.shape[1])
                result_df = pd.concat([result_df,row_data], ignore_index=True)
                index += 1
    return result_df

"""_summary_transformercsv：①Device-wise operation; ②Device fusion operation
folder_path
new_folder = r'D:\projects\Har-datasets\thidssets-nine\0.subsamples_win300\sixaxis'
csv_patterns = {
    "_jump.csv": 0,
    "_run.csv": 1,
    "_sit.csv": 2,
    "_stairdown.csv": 3,
    "_stairup.csv": 4,
    "_stand.csv": 5,
    "_walk.csv": 6,}
keep_columns = [0, 1, 2, 3, 4, 5]
keep_columns = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
"""
def transformercsv_deviceSplit(csv_patterns,folder_path,new_folder,windows_width,step,keep_columns):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder in subfolders:
        print(subfolder)
        result_df = transformcsv(csv_patterns, subfolder, windows_width, step,keep_columns)
        devicename = os.path.basename(os.path.normpath(subfolder))
        new_file_name = f"{devicename}_6axis_time_win_300.csv"
        new_file_path = os.path.join(new_folder, new_file_name)
        result_df.to_csv(new_file_path, index=False, header=False)

"""_summary_transformercsv使用实例：①分设备操作;②设备融合操作
    new_file_path:完整的新窗口文件的路径
"""
def transformercsv_deviceMerge(csv_patterns,folder_path,new_file_path,windows_width,step,keep_columns):
    result_df = transformcsv(csv_patterns, folder_path, windows_width, step,keep_columns)
    result_df.to_csv(new_file_path, index=False, header=False)

"""_summary_删除ku-Har窗口数据中异常数据
win_file_path:win数据的完整路径
newwin_file_path:新窗口数据的完整路径
"""
def cut_win_abnormal(win_file_path,newwin_file_path):
    df = pd.read_csv(win_file_path, header=None, index_col = None)
    df_array = np.array(df)
    indexes = [6587,6588,6589,6590,6591,6592,6593,6594,6595,6596,6597,6598,6599,6600,6601,6602,6603,
            6604,6605,6606,6607,
            6660,6661,6662,6663,6664,6665,6666,6667,6668,6669,6670,6671,6672,6673,6674,6675,6676,
            6677,6678,6679,6680,6681,6682,6683,6684,6685,6686,6687,
            6716,6717,6718,6719,6720,6721,6722,6723,6724,6725,6726,6727,6728,6729,6730,6731,6732,
            6733,6734,6735,6736,6737,6738,6739,6740,6741,6742,6743,
            6750,6751,6752,6753,6754,6755,6756,6757,6758,6759,6760,6761,6762,6763,6764,6765,6766,6767,]
    df_drop = np.delete(df_array, indexes, 0)
    df_drop_df = pd.DataFrame(df_drop)
    with open(newwin_file_path, 'w') as file:
        df_drop_df.to_csv(newwin_file_path, header = False, index=False)

"""Convert window data into training data for Random Forest
orignal_file_path
return data_x,data_y,file_name: pkl
"""
def win_to_RFdata(orignal_file_path,win):
    df = pd.read_csv(orignal_file_path, header=None)
    signals_colnums = df.shape[1]-3
    signals = df.values[:, 0:signals_colnums]
    signals = np.array(signals, dtype=np.float32)
    labels_index = df.shape[1]-3
    labels = df.values[:, labels_index]
    labels = np.array(labels, dtype=np.int64)
    signals = np.stack(
        [
            signals[:, i:i+win] for i in range(0, signals_colnums, win)
        ],
        axis=-1,
    )
    data_x = np.vstack(signals)
    labels_repeat = np.repeat(labels, win).reshape((signals.shape[0], win))
    labels_stack = np.stack(
        [
            labels_repeat[:, i:i+win] for i in range(0, win, win)
        ],
        axis=-1,
    )
    data_y = np.vstack(labels_stack)
    file_name = os.path.splitext(os.path.basename(orignal_file_path))[0]
    return data_x,data_y,file_name

"""Save the return value of win_to_RFdata to a pickle file
    tar_dir: eg, tar_dir = r"D:\projects\Har-datasets\thidssets-nine\0.RF_datasets"
    """
def save_to_pkl(data_x,data_y, tar_dir,file_name):
    obj = (data_x, data_y)
    target_filename = os.path.join(tar_dir, f'{file_name}.pkl')
    with open(target_filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

"""Load data from the pickle file
file_path: Full path of the pickle file
return data_x, data_y
"""
def load_dataset_pkl(file_path):
    with open(file_path, 'rb') as f:
        data_x, data_y, data_group = pickle.load(f)
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.uint8)
    data_group = data_group.astype(np.uint8)
    return data_x, data_y, data_group

"""
Convert the pkl data into NumPy compressed npz format.：After loading and splitting the pickle data, convert the training, test, and validation sets into a format suitable for Transformer training.
"""
def pkl_to_npz(data_x, data_y, data_group, win, step):
    unique_persons = set(data_group)
    unique_activities = set(data_y)

    data_x_window = []
    data_y_window = []
    data_group_window = []
    total_len = 0
    for person_id in unique_persons:
        for activity_id in unique_activities:
            person_activity_mask = (data_group == person_id) & (data_y == activity_id)
            person_activity_data = data_x[person_activity_mask]
            total_len = total_len + len(person_activity_data)
            for i in range(0, len(person_activity_data) - win + 1, step):
                window = person_activity_data[i:i + win]
                data_x_window.append(window)
                data_y_window.append(np.repeat(activity_id, win, axis=0))
                data_group_window.append(np.repeat(person_id, win, axis=0))
        
    signals = np.array(data_x_window, dtype=np.float32)
    act_labels = np.array(data_y_window, dtype=np.int64)
    person_labels = np.array(data_group_window, dtype=np.int64)
    return signals, act_labels, person_labels

""" the number of instances where the return value data_y from win_to_RFdata/load_dataset ?= a certain category
    actid:eg, actid=0
    win: The window size of the RF is set to 98
"""
def find_ydata_nums(data_y,actid,win):
    df = pd.DataFrame(data_y)
    num_zeros = df[df.iloc[:,0] == 1].shape[0]


"""_summary_Process abnormal data via win.csv: After using win_to_RFdata() to obtain data_x and data_y, process them with this method.
    data_x\data_y: Processed and obtained via win_to_RFdata()
    out_path: 3.1merge_sameActivity'
    group_names = {
        '0': '0.Stand',
        '1': '1.Sit',
        '2': '2.Talk-sit',
        '3': '3.Talk-stand',
        '4': '4.Stand-sit',
        '5': '5.Lay',
        '6': '6.Lay-stand',
        '7': '7.Pick',
        '8': '8.Jump',
        '9': '9.Push-up',
        '10': '10.Sit-up',
        '11': '11.Walk',
        '12': '12.Walk-backwards',
        '13': '13.Walk-circle',
        '14': '14.Run',
        '15': '15.Stair-up',
        '16': '16.Stair-down',
        '17': '17.Table-tennis'}
"""
def win_topkl_analy(data_x,data_y,out_path,group_names):
    df_x = pd.DataFrame(data_x)
    df_y = pd.DataFrame(data_y)
    df = pd.concat([df_x, df_y], axis=1)
    df.columns = ['0', '1', '2', '3', '4', '5', '6']
    df.iloc[:, 6] = df.iloc[:, 6].astype(str)
    grouped = df.groupby(df.columns[6])
    for name, group in grouped:
        out_file_name = group_names[str(name)] + '.csv'
        out_file_path = os.path.join(out_path, out_file_name)
        group = group.drop(group.columns[6], axis=1)
        try: 
            f = open(out_file_path, 'w+') 
            if f: 
                f.truncate()
            group.to_csv(out_file_path, index=False, header=False)
        except UnicodeEncodeError: 
            print("Encoding error. The data cannot be written to the file, so it is directly ignored")
