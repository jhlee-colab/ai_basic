import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer

class AutoEncoder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self._build_model()
    
    def _build_model(self):
        encoder_input = Input(shape=(784, ))
        encoded = Dense(392, activation='relu')(encoder_input)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(self.cfg['latent_dim'], activation='relu')(encoded)
        encoder = Model(encoder_input, encoded)
        self.encoder = encoder
        
        decoder_input = Input(shape=(self.cfg['latent_dim'],))
        decoded = Dense(392, activation='relu')(decoder_input)
        decoded = Dropout(0.1)(decoded)
        decoded = Dense(784, activation='sigmoid')(decoded)
        decoder = Model(decoder_input, decoded)
        self.decoder = decoder
        
        self.autoencoder = Model(encoder_input, decoder(encoder(encoder_input)))
        print('SimpleAutoEncoder, Model summary: ', self.autoencoder.summary())
        
    def compile(self, **kwargs):
        self.autoencoder.compile(**kwargs)
        
    def fit(self, **kwargs):
        return self.autoencoder.fit(**kwargs)
        
    def predict(self, **kwargs):
        return self.autoencoder.predict(**kwargs)
    
    def get_x_latent(self, **kwargs):
        return self.encoder.predict(**kwargs)