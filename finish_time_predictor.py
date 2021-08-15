import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import process_one_dim_to_two_dim_sec, seconds_to_string

MILESTONE = [
    "5Km", "10Km", "15Km", "20Km", "Half", "25Km",
    "30Km", "35Km", "40Km", "Finish"]

MILESTONE_FLOAT = np.array(
    [5, 10, 15, 20, 42.195 / 2, 25, 30, 35, 40, 42.195]) / 42.195


class Encoder(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.dense = tf.keras.layers.Dense(args.hidden_size)
        self.cell = tf.keras.layers.GRUCell(args.hidden_size)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, kmforenc, args, training):
        zero = tf.zeros((x.shape[0], args.hidden_size))
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
    def __init__(self, args):
        import tensorflow_probability as tfp
        super().__init__()
        self.cell = tf.keras.layers.GRUCell(args.hidden_size)
        self.dense_0 = tf.keras.layers.Dense(
            args.hidden_size / 2, activation="relu")
        self.dense = tf.keras.layers.Dense(2)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.posemb = tf.keras.layers.Embedding(10, args.hidden_size)
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

        for i in range(10 - num_splits_recorded):
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


class FinishTimePredictor():

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def load_weights(self, args):
        self.encoder.load_weights(args.encoder_model_path)
        self.decoder.load_weights(args.decoder_model_path)

    def train(self, dataset, eval_dataset, args):
        optimizer = tf.optimizers.Adam(learning_rate=0.0001)

        def negloglik(y, rv_y):
            return -rv_y.log_prob(tf.cast(y, tf.float32))
        patience = 0
        for i in tqdm(range(200)):
            for data in tqdm(dataset):
                enc_dec_splitid = np.random.randint(1, 10 - 1)
                dataforenc = data[:, :enc_dec_splitid]
                datafordec = data[:, enc_dec_splitid:]
                kmforenc = MILESTONE_FLOAT[:enc_dec_splitid]
                kmfordec = MILESTONE_FLOAT[enc_dec_splitid:]
                with tf.GradientTape() as tape:
                    splits_recorded, state = self.encoder(dataforenc,
                                                          kmforenc, args,
                                                          training=True)
                    predicts = self.decoder(state, splits_recorded,
                                            enc_dec_splitid, kmfordec,
                                            dataforenc,
                                            training=True)
                    loss = negloglik(datafordec, predicts)
                grads = tape.gradient(
                    loss,
                    self.encoder.trainable_weights +
                    self.decoder.trainable_weights)
                optimizer.apply_gradients(
                    zip(
                        grads,
                        self.encoder.trainable_weights +
                        self.decoder.trainable_weights))
            test_losses = []
            for enc_dec_splitid in range(1, 10):
                for test_data in eval_dataset:
                    dataforenc = test_data[:, :enc_dec_splitid]
                    datafordec = test_data[:, enc_dec_splitid:]
                    kmforenc = MILESTONE_FLOAT[:enc_dec_splitid]
                    kmfordec = MILESTONE_FLOAT[enc_dec_splitid:]
                    splits_recorded, state = self.encoder(dataforenc,
                                                          kmforenc, args,
                                                          training=False)
                    predicts = self.decoder(
                        state, splits_recorded, enc_dec_splitid, kmfordec,
                        dataforenc, training=False)
                    loss = negloglik(datafordec, predicts)
                    test_losses.append(loss)
            print("\n", np.mean(test_losses))
            if i == 0:
                best = np.mean(test_losses)
                self.encoder.save_weights(args.encoder_model_path)
                self.decoder.save_weights(args.decoder_model_path)
            else:
                if np.mean(test_losses) > best:
                    if patience == 5:
                        print("Early stopping.")
                        return
                    else:
                        patience += 1
                else:
                    best = min(best, np.mean(test_losses))
                    self.encoder.save_weights(args.encoder_model_path)
                    self.decoder.save_weights(args.decoder_model_path)
                    patience = 0

    def validate(self, dataset, args):
        for data in dataset:
            assert data.shape[0] == 1
            enc_dec_splitid = np.random.randint(1, 9)
            self.get_sample(data, enc_dec_splitid, args)
            one_dim_encoder_data = data[0, :enc_dec_splitid]
            one_dim_decoder_data = data[0, enc_dec_splitid:]
            self.print_estimation(one_dim_encoder_data, one_dim_decoder_data)
            graph = Graph(self)
            graph.add_actual_data(one_dim_encoder_data, one_dim_decoder_data)
            graph.add_estimated_data()
            graph.show()

    def predict(self, one_dim_baseline, args, one_dim_whatif=None):
        enc_dec_splitid = len(one_dim_baseline)
        two_dim_list = process_one_dim_to_two_dim_sec(one_dim_baseline)
        self.get_sample(two_dim_list, enc_dec_splitid, args)
        one_dim_encoder_data = one_dim_baseline[:enc_dec_splitid]
        self.print_estimation(one_dim_encoder_data)
        graph = Graph(self)
        # graph.add_actual_data(one_dim_encoder_data)
        graph.add_estimated_data()

        if one_dim_whatif is not None:
            enc_dec_splitid = len(one_dim_whatif)
            two_dim_list = process_one_dim_to_two_dim_sec(one_dim_whatif)
            self.get_sample(two_dim_list, enc_dec_splitid, args)
            one_dim_encoder_data = one_dim_whatif[:enc_dec_splitid]
            self.print_estimation(one_dim_encoder_data)
            # graph.add_actual_data(one_dim_encoder_data)
            graph.add_estimated_data(color="green")
        graph.save(args)

    def get_sample(self, two_dim_list, enc_dec_splitid, args):
        dataforenc = two_dim_list[:, :enc_dec_splitid]

        kmforenc = MILESTONE_FLOAT[:enc_dec_splitid]
        kmfordec = MILESTONE_FLOAT[enc_dec_splitid:]
        splits_recorded, state = self.encoder(
            dataforenc, kmforenc, args, training=False)
        predict_dist = self.decoder(state, splits_recorded,
                                    enc_dec_splitid, kmfordec,
                                    dataforenc, training=False)
        sample = predict_dist.sample(100000)
        self.upper_95 = np.percentile(sample, 97.5, 0)
        self.lower_95 = np.percentile(sample, 2.5, 0)
        self.middle = np.percentile(sample, 50, 0)
        self.upper_50 = np.percentile(sample, 75, 0)
        self.lower_50 = np.percentile(sample, 25, 0)

    def print_estimation(self, one_dim_encoder_data,
                         one_dim_decoder_data=None, batch_idx=0):
        print("**Estimation**")
        for i in range(len(MILESTONE)):
            kmidx = MILESTONE[i]
            if i < len(one_dim_encoder_data):
                actual_data = one_dim_encoder_data[i].ljust(10)
                estimated_data = \
                    "lower_95 => ******* lower_50 => ******* median => ******* " \
                    "upper_50 => ******* upper_95 => *******"
            else:
                if one_dim_decoder_data is not None:
                    actual_data = \
                        one_dim_decoder_data[i - len(one_dim_encoder_data)]
                    actual_data = actual_data.ljust(10)
                else:
                    actual_data = "".ljust(10)
                lower_95 = seconds_to_string(
                    self.lower_95[batch_idx, i - len(one_dim_encoder_data)])
                lower_50 = seconds_to_string(
                    self.lower_50[batch_idx, i - len(one_dim_encoder_data)])
                middle = seconds_to_string(
                    self.middle[batch_idx, i - len(one_dim_encoder_data)])
                upper_50 = seconds_to_string(
                    self.upper_50[batch_idx, i - len(one_dim_encoder_data)])
                upper_95 = seconds_to_string(
                    self.upper_95[batch_idx, i - len(one_dim_encoder_data)])
                estimated_data = \
                    f"lower_95 => {lower_95} " \
                    f"lower_50 => {lower_50} " \
                    f"median => {middle} " \
                    f"upper_50 => {upper_50} " \
                    f"upper_95 => {upper_95} "
            print(
                kmidx.ljust(10),
                actual_data, estimated_data
            )


class Graph():

    def __init__(self, finish_time_predictor):
        self.finish_time_predictor = finish_time_predictor
        self.fig, self.ax = plt.subplots()
        self.xaxis = [5, 10, 15, 20, 42.195 / 2, 25, 30, 35, 40, 42.195]
        self.ax.set_xlim(-1, 45)
        self.ax.set_xlabel('Kilometer')
        self.ax.set_ylabel('Hours')
        self.ax.set_title("Finish time prediction")
        # self.ax.margins(x=5)

    def add_actual_data(self,
                        one_dim_encoder_data,
                        one_dim_decoder_data=None):
        one_dim_encoder_data = \
            process_one_dim_to_two_dim_sec(one_dim_encoder_data)[0]
        self.ax.plot(
            self.xaxis[:len(one_dim_encoder_data)],
            one_dim_encoder_data, linestyle='--', marker='o')
        if one_dim_decoder_data is not None:
            one_dim_decoder_data = \
                process_one_dim_to_two_dim_sec(one_dim_decoder_data)[0]
            self.ax.plot(
                self.xaxis[len(one_dim_encoder_data):],
                one_dim_decoder_data, linestyle='--', marker='o')

    def add_estimated_data(self, batch_idx=0, color="orange"):
        prediction_steps = self.finish_time_predictor.middle.shape[-1]
        self.ax.plot(
            self.xaxis[-prediction_steps:],
            self.finish_time_predictor.middle[batch_idx], label="forecast",
            linestyle='--', marker='o', color=color)
        self.ax.fill_between(
            self.xaxis[-prediction_steps:],
            self.finish_time_predictor.lower_95[batch_idx],
            self.finish_time_predictor.upper_95[batch_idx],
            color=color, alpha=0.3, label="ci95%")
        self.ax.fill_between(
            self.xaxis[-prediction_steps:],
            self.finish_time_predictor.lower_50[batch_idx],
            self.finish_time_predictor.upper_50[batch_idx],
            color=color, alpha=0.5, label="ci50%")

    def save(self, args):
        self.ax.legend()
        self.fig.savefig(args.graph_save_path)

    def show(self):
        self.ax.legend()
        self.fig.show()
