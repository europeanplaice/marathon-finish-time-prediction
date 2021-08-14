import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from utils import parse_time
sns.set(font="ricty diminished")
HIDDEN_SIZE = 128
BATCH_SIZE = 256
num_splits = 10
km = np.array([5, 10, 15, 20, 42.195 / 2, 25, 30, 35, 40, 42.195]) / 42.195


class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(HIDDEN_SIZE)
        self.cell = tf.keras.layers.GRUCell(HIDDEN_SIZE)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, kmforenc, training):
        zero = tf.zeros((x.shape[0], HIDDEN_SIZE))
        for i in range(x.shape[1]):
            x /= kmforenc[i]
            x = self.dense(tf.expand_dims(x[:, i], 1))
            if i == 0:
                x, state = self.cell(x, states=[zero])
            else:
                x, state = self.cell(x, states=state)
        # x = self.dropout(x, training=training)
        return x, state


class Decoder(tf.keras.Model):
    def __init__(self):
        import tensorflow_probability as tfp
        super().__init__()
        self.cell = tf.keras.layers.GRUCell(HIDDEN_SIZE)
        self.dense_0 = tf.keras.layers.Dense(
            HIDDEN_SIZE / 2, activation="relu")
        self.dense = tf.keras.layers.Dense(2)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.posemb = tf.keras.layers.Embedding(num_splits, HIDDEN_SIZE)
        self.prob = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Independent(tfp.distributions.Normal(
                loc=t[:, :, 0],
                scale=tf.math.softplus(t[:, :, 1]),
            ),
                reinterpreted_batch_ndims=2
            )
        )

    def call(
            self, state, last_splits_recorded, num_splits_recorded, kmfordec,
            dataforenc, training):
        predicts = []

        for i in range(num_splits - num_splits_recorded):
            pos = tf.ones((last_splits_recorded.shape[0], 1))
            pos = pos * (num_splits_recorded + i)
            posencoded = tf.squeeze(self.posemb(pos), 1)
            if i == 0:
                last_splits_recorded += posencoded
                x, state = self.cell(last_splits_recorded, states=state)
            else:
                x += posencoded
                x, state = self.cell(x, states=state)
            pred = self.dense(self.dense_0(x))
            mean = tf.cast(dataforenc[:, -1], tf.float32)
            mean = mean + pred[:, 0] * kmfordec[i]
            std = pred[:, 1]
            pred = tf.stack([mean, std], 1)
            predicts.append(pred)
        predicts = tf.stack(predicts, 1)
        dist = self.prob(predicts)
        return dist


def train(dataset, test_dataset, args):
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    encoder = Encoder()
    decoder = Decoder()

    def negloglik(y, rv_y):
        return -rv_y.log_prob(tf.cast(y, tf.float32))
    patience = 0
    for i in tqdm(range(200)):
        for data in dataset:
            enc_dec_splitid = np.random.randint(1, num_splits - 1)
            dataforenc = data[:, :enc_dec_splitid]
            datafordec = data[:, enc_dec_splitid:]
            kmforenc = km[:enc_dec_splitid]
            kmfordec = km[enc_dec_splitid:]
            with tf.GradientTape() as tape:
                splits_recorded, state = encoder(dataforenc, kmforenc,
                                                 training=True)
                predicts = decoder(state, splits_recorded,
                                   enc_dec_splitid, kmfordec, dataforenc,
                                   training=True)
                loss = negloglik(datafordec, predicts)
            grads = tape.gradient(
                loss,
                encoder.trainable_weights + decoder.trainable_weights)
            optimizer.apply_gradients(
                zip(
                    grads,
                    encoder.trainable_weights + decoder.trainable_weights))
        test_losses = []
        for enc_dec_splitid in range(1, num_splits):
            for test_data in test_dataset:
                dataforenc = test_data[:, :enc_dec_splitid]
                datafordec = test_data[:, enc_dec_splitid:]
                kmforenc = km[:enc_dec_splitid]
                kmfordec = km[enc_dec_splitid:]
                splits_recorded, state = encoder(dataforenc, kmforenc,
                                                 training=False)
                predicts = decoder(
                    state, splits_recorded, enc_dec_splitid, kmfordec,
                    dataforenc, training=False)
                loss = negloglik(datafordec, predicts)

                test_losses.append(loss)
        print("\n", np.mean(test_losses))
        if i == 0:
            best = np.mean(test_losses)
            encoder.save_weights(args.encoder_model_path)
            decoder.save_weights(args.decoder_model_path)
        else:
            if np.mean(test_losses) > best:
                if patience == 5:
                    print("Early stopping.")
                    break
                else:
                    patience += 1
            else:
                best = min(best, np.mean(test_losses))
                encoder.save_weights(args.encoder_model_path)
                decoder.save_weights(args.decoder_model_path)
                patience = 0
    return encoder, decoder


def makedataset(data):
    idx = int(len(data) * 0.8)
    dataset = tf.data.Dataset.from_tensor_slices((data[:idx]))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(BATCH_SIZE)
    _valid_dataset = tf.data.Dataset.from_tensor_slices((data[idx:]))
    _valid_dataset = _valid_dataset.shuffle(buffer_size=100)
    valid_dataset = _valid_dataset.batch(BATCH_SIZE)
    test_dataset = _valid_dataset.batch(1)
    return dataset, valid_dataset, test_dataset


def predict_from_record(data, encoder, decoder, enc_dec_splitid):
    dataforenc = data[:, :enc_dec_splitid]

    kmforenc = km[:enc_dec_splitid]
    kmfordec = km[enc_dec_splitid:]
    splits_recorded, state = encoder(dataforenc, kmforenc, training=False)
    predict_dist = decoder(state, splits_recorded,
                           enc_dec_splitid, kmfordec,
                           dataforenc, training=False)
    sample = predict_dist.sample(100000)
    return sample


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


def predict(record_list, encoder, decoder):
    enc_dec_splitid = len(record_list)
    record_list = np.array([parse_time(time) for time in record_list])
    record_list /= (60 * 60)
    record_list = np.expand_dims(record_list, 0)
    sample = predict_from_record(
        record_list, encoder, decoder, enc_dec_splitid)
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
    parser.add_argument("--full_record")
    parser.add_argument("--encoder_model_path",
                        default='trained_model/encoder/encoder')
    parser.add_argument("--decoder_model_path",
                        default='trained_model/decoder/decoder')

    args = parser.parse_args()
    if args.elapsed_time is not None:
        args.do_predict = True

    if args.do_train or args.do_eval:

        df = pd.read_csv(args.train_data_path)
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

        data = df[
            ["5K", "10K", "15K", "20K", "Half",
             "25K", "30K", "35K", "40K", "Official Time"]]
        data = data[data.min(1) > 0.]
        data = data.values

        data /= (60 * 60)

        dataset, valid_dataset, test_dataset = makedataset(data)
        if args.do_train:
            encoder, decoder = train(dataset, valid_dataset, args)
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_weights(args.encoder_model_path)
    decoder.load_weights(args.decoder_model_path)
    if args.do_eval:
        validate(test_dataset, encoder, decoder)
    if args.do_predict:
        predict(args.elapsed_time.split(","), encoder, decoder)


if __name__ == '__main__':
    main()
