
from tensorflow.keras.layers import Dense, \
    GRU, Input, Bidirectional, RepeatVector, \
    TimeDistributed, Lambda
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

from util import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def build_model():


    encoder_input = Input(shape=(time_step, input_dim), name='encoder_input')

    rnn1 = Bidirectional(GRU(rnn_dim, return_sequences=True), name='rnn1')(encoder_input)
    rnn2 = Bidirectional(GRU(rnn_dim), name='rnn2')(rnn1)



    z_mean = Dense(z_dim, name='z_mean')(rnn2)
    z_log_var = Dense(z_dim, name='z_log_var')(rnn2)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(z_dim,), name='z')([z_mean, z_log_var])
    class kl_beta(tf.keras.layers.Layer):
        def __init__(self):
            super(kl_beta, self).__init__()

            # your variable goes here
            self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        def call(self, inputs, **kwargs):
            # your mul operation goes here
            return -self.beta *inputs

    beta = kl_beta()
    encoder = Model(encoder_input, z, name='encoder')

    # decoder

    decoder_latent_input = Input(shape=z_dim, name='z_sampling')

    repeated_z = RepeatVector(time_step, name='repeated_z_tension')(decoder_latent_input)



    rnn1_output = GRU(rnn_dim, name='decoder_rnn1', return_sequences=True)(repeated_z)

    rnn2_output = GRU(rnn_dim, name='decoder_rnn2', return_sequences=True)(
        rnn1_output)

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)

    kl_loss = 0.5 *kl_loss

    kl_loss = beta(kl_loss)
    tensile_middle_output = TimeDistributed(Dense(tension_middle_dim, activation='elu'),
                                            name='tensile_strain_dense1')(rnn2_output)

    tensile_output = TimeDistributed(Dense(tension_output_dim, activation='elu'),
                                     name='tensile_strain_dense2')(tensile_middle_output)

    diameter_middle_output = TimeDistributed(Dense(tension_middle_dim, activation='elu'),
                                             name='diameter_strain_dense1')(rnn2_output)

    diameter_output = TimeDistributed(Dense(tension_output_dim, activation='elu'),
                                      name='diameter_strain_dense2')(diameter_middle_output)

    melody_rhythm_1 = TimeDistributed(Dense(start_middle_dim, activation='elu'),
                                      name='melody_start_dense1')(rnn2_output)
    melody_rhythm_output = TimeDistributed(Dense(melody_note_start_dim, activation='sigmoid'),
                                           name='melody_start_dense2')(
        melody_rhythm_1)

    melody_pitch_1 = TimeDistributed(Dense(melody_bass_dense_1_dim, activation='elu'),
                                     name='melody_pitch_dense1')(rnn2_output)

    melody_pitch_output = TimeDistributed(Dense(melody_output_dim, activation='softmax'),
                                          name='melody_pitch_dense2')(melody_pitch_1)

    bass_rhythm_1 = TimeDistributed(Dense(start_middle_dim, activation='elu'),
                                    name='bass_start_dense1')(rnn2_output)

    bass_rhythm_output = TimeDistributed(Dense(bass_note_start_dim, activation='sigmoid'),
                                         name='bass_start_dense2')(
        bass_rhythm_1)

    bass_pitch_1 = TimeDistributed(Dense(melody_bass_dense_1_dim, activation='elu'),
                                   name='bass_pitch_dense1')(rnn2_output)
    bass_pitch_output = TimeDistributed(Dense(bass_output_dim, activation='softmax'),
                                        name='bass_pitch_dense2')(bass_pitch_1)

    decoder_output = [melody_pitch_output, melody_rhythm_output, bass_pitch_output, bass_rhythm_output,
                      tensile_output, diameter_output
                      ]

    decoder = Model(decoder_latent_input, decoder_output, name='decoder')

    model_input = encoder_input

    vae = Model(model_input, decoder(encoder(model_input)), name='encoder_decoder')

    vae.add_loss(kl_loss)

    vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    optimizer = keras.optimizers.Adam()



    vae.compile(optimizer=optimizer,
                loss=['categorical_crossentropy', 'binary_crossentropy',
                      'categorical_crossentropy', 'binary_crossentropy',
                      'mse', 'mse'
                      ],
                metrics=[[keras.metrics.CategoricalAccuracy()],
                         [keras.metrics.BinaryAccuracy()],
                         [keras.metrics.CategoricalAccuracy()],
                         [keras.metrics.BinaryAccuracy()],
                         [keras.metrics.MeanSquaredError()],
                         [keras.metrics.MeanSquaredError()]
                         ]
                )

    return vae


def draw_two_figure(tensile_strain, diameter, first_name='tensile strain',
                    second_name='diameter',
                    file_name='default.png', y_label='tension',
                    title='tension figure',
                    save=False):
    if tensile_strain.shape[0] == 64:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(tensile_strain, label=first_name)
        ax.plot(diameter, label=second_name)
        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel('timestep')
        ax.set_title(title)

    if save is True:
        plt.savefig(file_name)

    plt.show()
    plt.close('all')

def manipuate_latent_space(piano_roll, vector_up_t, vector_high_d, vector_up_down_t,
                                                 vae,t_up_factor,d_high_factor,t_up_down_factor,
                                                   change_t=True,change_d=False,change_t_up_down=False,
                                                   with_input=True,draw_tension=True):

    if with_input and piano_roll is not None:
        piano_roll = np.expand_dims(piano_roll, 0)
        z = vae.layers[1].predict(piano_roll)
    else:
        z = np.random.normal(size=(1,z_dim))

    reconstruction = vae.layers[2].predict(z)

    tensile_reconstruction = np.squeeze(reconstruction[-2])
    diameter_reconstruction = np.squeeze(reconstruction[-1])

    # recon_result = result_sampling(np.concatenate(list(reconstruction), axis=-1))[0]
    changed_z = z
    if change_t:
        changed_z += t_up_factor * vector_up_t

    if change_d:
        changed_z += d_high_factor * vector_high_d

    if change_t_up_down:
        changed_z += t_up_down_factor * vector_up_down_t

    changed_reconstruction = vae.layers[2].predict(changed_z)

    changed_recon_result = result_sampling(np.concatenate(list(changed_reconstruction), axis=-1))[0]

    changed_tensile_reconstruction = np.squeeze(changed_reconstruction[-2])

    changed_diameter_reconstruction = np.squeeze(changed_reconstruction[-1])

    if draw_tension:
        draw_two_figure(tensile_reconstruction,diameter_reconstruction,title='original tension')
        draw_two_figure(changed_tensile_reconstruction,changed_diameter_reconstruction,title='changed tension')


    return roll_to_pretty_midi(changed_recon_result)





