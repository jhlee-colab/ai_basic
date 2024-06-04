import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


class VAELossLayer(Layer):
    def __init__(self, beta=0.0005, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
    
    def calculate_loss(self, original_input, reconstructed_output, mu, sigma):
        original_input = K.flatten(original_input)
        reconstructed_output = K.flatten(reconstructed_output)
        reconstruction_loss = tf.keras.metrics.binary_crossentropy(original_input, reconstructed_output)
        kl_loss = -self.beta * K.mean(1 + sigma - K.square(mu) - K.exp(sigma), axis=-1)
        
        self.add_metric(reconstruction_loss, name='reconstruction_loss')
        self.add_metric(kl_loss, name='kl_loss')

        return K.mean(reconstruction_loss + kl_loss)

    def call(self, inputs):
        """
        Computes the loss and adds it to the layer's losses
        """
        original_input, reconstructed_output, mu, sigma = inputs
        loss = self.calculate_loss(original_input, reconstructed_output, mu, sigma)
        self.add_loss(loss, inputs=inputs)
        return original_input
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta
        })
        return config

class VariationalAutoEncoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self._build_model()
    
    def _build_model(self):
        # Encoder
        encoder_input = Input(shape=(28, 28, 1))
        encoded = Conv2D(32, 3, padding='same', activation='relu')(encoder_input)
        encoded = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(encoded)
        encoded = Conv2D(64, 3, padding='same', activation='relu')(encoded)
        encoded = Conv2D(64, 3, padding='same', activation='relu')(encoded)

        en_shape = K.int_shape(encoded)
        
        flatten = Flatten()(encoded)
        encoded = Dense(32, activation='relu')(flatten)  
        latent_mu = Dense(self.cfg['latent_dim'], name='latent_mu')(encoded)
        latent_sigma = Dense(self.cfg['latent_dim'], name='latent_sigma')(encoded)
        
        z = Lambda(self._sample_z, output_shape=(self.cfg['latent_dim'],), name='z')([latent_mu, latent_sigma])
        
        encoder = Model(encoder_input, [latent_mu, latent_sigma, z], name='encoder_model')
        self.encoder = encoder
        
        # Decoder
        decoder_input = Input(shape=(self.cfg['latent_dim'],), name='decoder_input')
        decoder_dense = Dense(en_shape[1] * en_shape[2] * en_shape[3], activation='relu')(decoder_input)
        decoder_reshape = Reshape((en_shape[1], en_shape[2], en_shape[3]))(decoder_dense)
        decoder_upsample = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(decoder_reshape)
        decoder_output = Conv2DTranspose(1, 3, padding='same', activation='sigmoid', name='decoder_output')(decoder_upsample)
        
        decoder = Model(decoder_input, decoder_output, name='decoder_model')
        self.decoder = decoder
        
        reconstructed_output = self.decoder(z)
        
        vae_loss = VAELossLayer()([encoder_input, reconstructed_output, latent_mu, latent_sigma])
        
        autoencoder = Model(encoder_input, vae_loss, name='vae_model')
        self.autoencoder = autoencoder
    
    def _sample_z(self, args):
        latent_mu, latent_sigma = args
        batch_size = K.shape(latent_mu)[0]
        dim = K.int_shape(latent_mu)[1]
        
        epsilon = K.random_normal(shape=(batch_size, dim))
        return latent_mu + K.exp(latent_sigma / 2) * epsilon
    
    def compile(self, **kwargs):
        self.autoencoder.compile(**kwargs)
        
    def fit(self, **kwargs):
        return self.autoencoder.fit(**kwargs)
    
    def predict(self, **kwargs):
        mu, sigma, z = self.encoder.predict(**kwargs)
        reconstructed_outputs = self.decoder.predict(z)
        return reconstructed_outputs
    
    def get_x_latent(self, **kwargs):
        mu, sigma, z = self.encoder.predict(**kwargs)
        return z
        return self.autoencoder.predict(**kwargs)
    
    