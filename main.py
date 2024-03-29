import argparse

import pandas as pd
import seaborn as sns

from finish_time_predictor import Decoder, Encoder, FinishTimePredictor
from utils import makedataset, preprocess_rawdata

sns.set(font="ricty diminished")


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_all_splits_eval", action="store_true")
    parser.add_argument("--train_data_path", default="boston2017-2018.csv")
    parser.add_argument("--elapsed_time")
    parser.add_argument("--elapsed_time_what_if")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--full_record")
    parser.add_argument("--graph_save_path", default="estimation.jpg")
    parser.add_argument("--encoder_model_path", default="trained_model/encoder/encoder")
    parser.add_argument("--decoder_model_path", default="trained_model/decoder/decoder")
    return parser


def main(args=None):

    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    if args.elapsed_time is not None:
        args.do_predict = True
    encoder = Encoder(args)
    decoder = Decoder(args)
    finish_time_predictor = FinishTimePredictor(encoder, decoder)
    if args.do_train or args.do_eval:

        df = pd.read_csv(args.train_data_path)
        # save_feather(df)
        data = preprocess_rawdata(df)

        train_dataset, eval_dataset, eval_onebatch_dataset = makedataset(
            data, args, True
        )
        if args.do_train:
            finish_time_predictor.train(train_dataset, eval_dataset, args)
        if args.do_eval:
            finish_time_predictor.load_weights(args)
            finish_time_predictor.validate(eval_onebatch_dataset, args)
    if args.do_predict:
        finish_time_predictor.load_weights(args)
        if args.elapsed_time_what_if is not None:
            finish_time_predictor.predict(
                args.elapsed_time.split(","),
                args,
                args.elapsed_time_what_if.split(","),
            )
        else:
            finish_time_predictor.predict(
                args.elapsed_time.split(","),
                args,
            )


if __name__ == "__main__":
    main()
