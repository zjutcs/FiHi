import numpy as np
import pandas as pd
def read_csv(file_path):
    df = pd.read_csv(file_path, header=0,index_col=False,encoding='utf-8')
    return df
 
def load_datasets(submerge_actclass_df):
    data = submerge_actclass_df.iloc[: , : submerge_actclass_df.shape[1]-2]
    labels = submerge_actclass_df.iloc[: , submerge_actclass_df.shape[1]-1]
    subject = submerge_actclass_df.iloc[: , submerge_actclass_df.shape[1]-2]

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    subject = np.array(subject, dtype=np.int64)
    return data, labels, subject

"""
Convert the data directly into 3D data in npz format
"""
def pkl_to_npz_new(data_x, data_y, data_group, train_group, test_group, win, step):
    #Remove repetition
    unique_persons = set(data_group)
    unique_activities = set(data_y)

    total_len = 0
    data_x_window = []
    data_y_window = []
    data_group_window = []
    
    total_len_test = 0
    data_x_window_test = []
    data_y_window_test = []
    data_group_window_test = []
    
    for person_id in unique_persons:
        if (train_group <= unique_persons) and (test_group <= unique_persons):
            if person_id in train_group:
                for activity_id in unique_activities:
                    person_activity_mask = (data_group == person_id) & (data_y == activity_id)
                    person_activity_data = data_x[person_activity_mask]

                    # divide the window dataset
                    total_len = total_len + len(person_activity_data)
                    winnum = 0
                    for i in range(0, len(person_activity_data) - win + 1, step):
                        window = person_activity_data[i:i + win]
                        data_x_window.append(window)
                        data_y_window.append(np.repeat(activity_id, win, axis=0))
                        data_group_window.append(np.repeat(person_id, win, axis=0))
                        winnum = winnum + 1
            # test sample
            if person_id in test_group:
                for activity_id in unique_activities:
                    person_activity_mask = (data_group == person_id) & (data_y == activity_id)
                    person_activity_data = data_x[person_activity_mask]
                    
                    """reset index"""
                    winnum_test = 0
                    total_len_test = total_len_test + len(person_activity_data)
                    for i in range(0, len(person_activity_data) - win + 1, step):
                        window = person_activity_data[i:i + win]
                        data_x_window_test.append(window)
                        data_y_window_test.append(np.repeat(activity_id, win, axis=0))
                        data_group_window_test.append(np.repeat(person_id, win, axis=0))
                        winnum_test = winnum_test + 1
        else:
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
    
    if (train_group <= unique_persons) and (test_group <= unique_persons):
        signals_test = np.array(data_x_window_test, dtype=np.float32)
        act_labels_test = np.array(data_y_window_test, dtype=np.int64)
        person_labels_test = np.array(data_group_window_test, dtype=np.int64)
        return signals, act_labels, person_labels,signals_test,act_labels_test,person_labels_test
    else:
        return signals, act_labels, person_labels

""" loso already exists:
Convert the data directly into 3D data in npz format
"""
def pkl_to_npz(data_x, data_y, data_group, Loso, win, step, group):

    unique_persons = set(data_group)
    unique_activities = set(data_y)

    total_len = 0
    data_x_window = []
    data_y_window = []
    data_group_window = []

    total_len_test = 0
    data_x_window_test = []
    data_y_window_test = []
    data_group_window_test = []
    

    for person_id in unique_persons:
        if Loso in group:
            if person_id != Loso:
                for activity_id in unique_activities:
                    person_activity_mask = (data_group == person_id) & (data_y == activity_id)
                    person_activity_data = data_x[person_activity_mask]
                    
                    """reset index"""
                    total_len = total_len + len(person_activity_data)
                    winnum = 0
                    for i in range(0, len(person_activity_data) - win + 1, step):
                        window = person_activity_data[i:i + win]
                        data_x_window.append(window)
                        data_y_window.append(np.repeat(activity_id, win, axis=0))
                        data_group_window.append(np.repeat(person_id, win, axis=0))
                        winnum = winnum + 1
            else:
                for activity_id in unique_activities:
                    person_activity_mask = (data_group == person_id) & (data_y == activity_id)
                    person_activity_data = data_x[person_activity_mask]
                    
                    """reset index"""
                    winnum_test = 0
                    total_len_test = total_len_test + len(person_activity_data)
                    for i in range(0, len(person_activity_data) - win + 1, step):
                        window = person_activity_data[i:i + win]
                        data_x_window_test.append(window)
                        data_y_window_test.append(np.repeat(activity_id, win, axis=0))
                        data_group_window_test.append(np.repeat(person_id, win, axis=0))
                        winnum_test = winnum_test + 1
        else:
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

    if Loso in group:
        signals_test = np.array(data_x_window_test, dtype=np.float32)
        act_labels_test = np.array(data_y_window_test, dtype=np.int64)
        person_labels_test = np.array(data_group_window_test, dtype=np.int64)
        return signals, act_labels, person_labels,signals_test,act_labels_test,person_labels_test
    else:
        return signals, act_labels, person_labels