import datetime

import numpy as np
import tensorflow as tf


def seconds_to_string(x):
    return str(datetime.timedelta(seconds=round(x * (60 * 60))))


def parse_time(x):
    try:
        splited = [int(y) for y in x.split(":")]
        time_parsed = datetime.time(*splited)
        time_parsed = datetime.timedelta(
            hours=time_parsed.hour,
            minutes=time_parsed.minute,
            seconds=time_parsed.second,
        )
        return time_parsed.total_seconds()
    except ValueError:
        return -1


def process_one_dim_to_two_dim_sec(one_dim_list):
    one_dim_list = np.array([parse_time(time) for time in one_dim_list])
    one_dim_list /= (60 * 60)
    one_dim_list = np.expand_dims(one_dim_list, 0)
    return one_dim_list


def preprocess_rawdata(df):

    df = df.sample(len(df))

    df["5K"] = df["5K"].apply(parse_time)
    df["10K"] = df["10K"].apply(parse_time)
    df["15K"] = df["15K"].apply(parse_time)
    df["20K"] = df["20K"].apply(parse_time)
    df["Half"] = df["Half"].apply(parse_time)
    df["25K"] = df["25K"].apply(parse_time)
    df["30K"] = df["30K"].apply(parse_time)
    df["35K"] = df["35K"].apply(parse_time)
    df["40K"] = df["40K"].apply(parse_time)
    df["Official Time"] = df["Official Time"].apply(parse_time)

    df = df[
        ["5K", "10K", "15K", "20K", "Half",
            "25K", "30K", "35K", "40K", "Official Time"]]
    df = df[df.min(1) > 0.]
    df = df.values

    df /= (60 * 60)
    return df


def get_tfdataset(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((data))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    return dataset


def makedataset(data, args, return_one_batch_eval_dataset):
    idx = int(len(data) * 0.8)
    train_dataset = get_tfdataset(data[:idx], args.batch_size)
    eval_dataset = get_tfdataset(data[idx:], args.batch_size)
    if return_one_batch_eval_dataset:
        test_dataset = get_tfdataset(data[idx:], 1)
        return train_dataset, eval_dataset, test_dataset
    else:
        return train_dataset, eval_dataset


def save_feather(df):
    df.to_feather("file.feather")
