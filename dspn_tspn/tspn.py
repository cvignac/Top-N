import tensorflow as tf
from models.fspool import FSEncoder
from models.transformer import Encoder
from models.set_prior import SetPrior, FirstKPrior, TopKPrior, MLPPrior
from models.size_predictor import SizePredictor


class Tspn(tf.keras.Model):
    def __init__(self, encoder_latent, encoder_out, fspool_n_pieces, transformer_layers, transformer_attn_size,
                 transformer_num_heads, num_element_features, size_pred_width, pad_value, max_set_size, set_generator):
        super(Tspn, self).__init__()

        self.pad_value = pad_value                                  # -1
        self.max_set_size = max_set_size                            # 360
        self.num_element_features = num_element_features            # 2 for mnist

        # TODO TUNE
        num_input_features = 16
        if set_generator == 'random':
            self._prior = SetPrior(num_input_features)
            print("Random generator")
        elif set_generator == 'mlp':
            self._prior = MLPPrior(num_input_features)
        elif set_generator == 'first':
            print("First N  generator")
            self._prior = FirstKPrior(num_input_features)
        elif set_generator == 'top':
            self._prior = TopKPrior(num_input_features)
            print("Top N generator")
        else:
            raise ValueError('Set generator unknown or not specified')

        self._encoder = FSEncoder(encoder_latent, encoder_out, fspool_n_pieces)

        self._transformer = Encoder(transformer_layers, transformer_attn_size, transformer_num_heads)
        # 256 channels in the output

        # initialise the output to predict points at the center of our canvas
        # num element features = 2
        self._set_prediction = tf.keras.layers.Conv1D(num_element_features, 1, kernel_initializer='zeros',
                                                      bias_initializer=tf.keras.initializers.constant(0.5),
                                                      use_bias=True)

        self._size_predictor = SizePredictor(size_pred_width, max_set_size)

    def call(self, initial_set, sizes):
        """ One forward pass."""
        # encode the input set
        encoded = self._encoder(initial_set, sizes)  # pooled: [batch_size, num_features]       32 x 64
        batch_size = tf.shape(encoded)[0]
        sampled_set = self.sample_prior_batch(encoded, batch_size)   # Returns a padded tensor    bs, n_max, set_channels
        # concat the encoded set vector onto each initial set element
        # TODO: change_film here
        encoded_shaped = tf.tile(tf.expand_dims(encoded, 1), [1, self.max_set_size, 1])
        sampled_elements_conditioned = tf.concat([sampled_set, encoded_shaped], 2)      # bs, n_max, dim_tot

        masked_values = tf.cast(tf.math.logical_not(tf.sequence_mask(sizes, self.max_set_size)), tf.float32)
        pred_set_latent = self._transformer(sampled_elements_conditioned, masked_values)

        pred_set = self._set_prediction(pred_set_latent)
        return pred_set, sampled_set

    def sample_prior(self, latent, batch_size):
        # total_elements = tf.reduce_sum(sizes)
        # TODO: the set generator is called here
        sampled_elements = self._prior(latent, batch_size)  # [total_set_size, num_features]
        return sampled_elements

    def sample_prior_batch(self, latent, batch_size):
        """ Given the latent vector for each batch element and the size of each set, return the initial sets"""
        sampled_elements = self.sample_prior(latent, batch_size)
        # samples_ragged = tf.RaggedTensor.from_row_lengths(sampled_elements, sizes)      # Tensor w varying length
        # padded_samples = samples_ragged.to_tensor(default_value=self.pad_value,
        #                                           shape=[sizes.shape[0], self.max_set_size, self.num_element_features])
        # return padded_samples
        return sampled_elements

    def encode_set(self, initial_set, sizes):
        return self._encoder(initial_set, sizes)

    def predict_size(self, embedding):
        sizes = self._size_predictor(embedding)
        sizes = tf.keras.activations.softmax(sizes, -1)
        return sizes

    def get_autoencoder_weights(self):
        return self._encoder.trainable_weights + \
               self._transformer.trainable_weights + \
               self._set_prediction.trainable_weights + \
               self._prior.trainable_weights

    def get_prior_weights(self):
        return self._prior.trainable_weights

    def get_size_predictor_weights(self):
        return self._size_predictor.trainable_weights
