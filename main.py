import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from utils import parse_time, preprocess_rawdata, makedataset
from utils import process_one_dim_to_two_dim_sec
from finish_time_predictor import Encoder, Decoder, FinishTimePredictor
sns.set(font="ricty diminished")
HIDDEN_SIZE = 128
BATCH_SIZE = 256
num_splits = 10
km = np.array([5, 10, 15, 20, 42.195 / 2, 25, 30, 35, 40, 42.195]) / 42.195


def predict_from_record(data, encoder, decoder, enc_dec_splitid):
    dataforenc = data[:, :enc_dec_splitid]

    kmforenc = km[:enc_dec_splitid]
    kmfordec = km[enc_dec_splitid:]
    splits_recorded, state = encoder(dataforenc, kmforenc, training=False)
    predict_dist = decoder(state, splits_recorded,
                           enc_dec_splitid, kmfordec,
                           dataforenc, training=False)
    sample = predict_dist.sample(100000)
    upper_95 = np.percentile(sample, 95, 0)
    lower_95 = np.percentile(sample, 5, 0)
    middle = np.percentile(sample, 50, 0)
    upper_50 = np.percentile(sample, 75, 0)
    lower_50 = np.percentile(sample, 25, 0)
    return sample, upper_95, lower_95, middle, upper_50, lower_50,


def draw_graph(sample, enc_dec_splitid, data=None):
    upper_95 = np.percentile(sample, 95, 0)
    lower_95 = np.percentile(sample, 5, 0)
    middle = np.percentile(sample, 50, 0)
    upper_50 = np.percentile(sample, 75, 0)
    lower_50 = np.percentile(sample, 25, 0)
    xaxis = [5, 10, 15, 20, 42.195 / 2, 25, 30, 35, 40, 42.195]
    if data is not None:
        dataforenc = data[:, :enc_dec_splitid]
        datafordec = data[:, enc_dec_splitid:]
        plt.plot(xaxis[:enc_dec_splitid], dataforenc[0],
                 linestyle='--', marker='o')
        plt.plot(xaxis[enc_dec_splitid:], datafordec[0], label="true",
                 linestyle='--', marker='o')
    plt.plot(xaxis[enc_dec_splitid:], middle[0], label="forecast",
             linestyle='--', marker='o')
    plt.fill_between(
        xaxis[enc_dec_splitid:],
        lower_95[0],
        upper_95[0], color='orange', alpha=0.3, label="ci95%")
    plt.fill_between(
        xaxis[enc_dec_splitid:],
        lower_50[0],
        upper_50[0], color='orange', alpha=0.5, label="ci50%")
    plt.xlabel('Kilometer')
    plt.ylabel('Hours')
    plt.title("Finish time prediction")
    plt.legend()
    plt.show()


def print_estimation(sample, enc_dec_splitid, data=None):
    if data is not None:
        datafordec = data[:, enc_dec_splitid:]
    upper_95 = np.percentile(sample, 97.5, 0)
    lower_95 = np.percentile(sample, 2.5, 0)
    middle = np.percentile(sample, 50, 0)
    upper_50 = np.percentile(sample, 75, 0)
    lower_50 = np.percentile(sample, 25, 0)
    _km = [
        "5Km", "10Km", "15Km", "20Km", "Half", "25Km",
        "30Km", "35Km", "40Km", "Finish"]
    print("**Estimation**")
    for i in range(sample.shape[2]):
        kmidx = _km[enc_dec_splitid + i]
        if data is not None:
            actual_data = np.array(datafordec[0])[i]
            actual_data = str(
                datetime.timedelta(seconds=round(actual_data * (60 * 60))))
            actual_data = f"(actual time:{actual_data}) "
        else:
            actual_data = ""
        print(
            kmidx.ljust(10),
            actual_data,
            "lower_95 =>",
            str(datetime.timedelta(seconds=round(lower_95[0, i] * (60 * 60)))),
            "  lower_50 =>",
            str(datetime.timedelta(seconds=round(lower_50[0, i] * (60 * 60)))),
            "  median =>",
            str(datetime.timedelta(seconds=round(middle[0, i] * (60 * 60)))),
            "  upper_50 =>",
            str(datetime.timedelta(seconds=round(upper_50[0, i] * (60 * 60)))),
            "  upper_95 =>",
            str(datetime.timedelta(seconds=round(upper_95[0, i] * (60 * 60)))),
        )


def validate(dataset, encoder, decoder):
    for data in dataset:
        data = data[:1]
        enc_dec_splitid = np.random.randint(1, num_splits - 1)
        sample = predict_from_record(data, encoder, decoder, enc_dec_splitid)
        print_estimation(sample, enc_dec_splitid, data)
        draw_graph(sample, enc_dec_splitid, data)


def predict(one_dim_list, encoder, decoder):
    enc_dec_splitid = len(one_dim_list)
    two_dim_list = process_one_dim_to_two_dim_sec(one_dim_list)
    sample = predict_from_record(
        two_dim_list, encoder, decoder, enc_dec_splitid)
    print_estimation(sample, enc_dec_splitid)
    draw_graph(sample, enc_dec_splitid)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_all_splits_eval", action="store_true")
    parser.add_argument("--train_data_path", default='boston2017-2018.csv')
    parser.add_argument("--elapsed_time")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--full_record")
    parser.add_argument("--graph_save_path", default="estimation.jpg")
    parser.add_argument("--encoder_model_path",
                        default='trained_model/encoder/encoder')
    parser.add_argument("--decoder_model_path",
                        default='trained_model/decoder/decoder')

    args = parser.parse_args()
    if args.elapsed_time is not None:
        args.do_predict = True
    encoder = Encoder(args)
    decoder = Decoder(args)
    finish_time_predictor = FinishTimePredictor(encoder, decoder)
    if args.do_train or args.do_eval:

        df = pd.read_csv(args.train_data_path)
        data = preprocess_rawdata(df)

        train_dataset, eval_dataset, eval_onebatch_dataset = \
            makedataset(data, args, True)
        if args.do_train:
            encoder, decoder = finish_time_predictor.train(
                train_dataset, eval_dataset, args)
    if args.do_eval:
        finish_time_predictor.load_weights(args)
        finish_time_predictor.validate(eval_onebatch_dataset, args)
    if args.do_predict:
        finish_time_predictor.load_weights(args)
        finish_time_predictor.predict(args.elapsed_time.split(","), args)


if __name__ == '__main__':
    main()
