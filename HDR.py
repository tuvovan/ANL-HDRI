import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.layers import Conv2D, LeakyReLU, ReLU, BatchNormalization, Conv2DTranspose, Reshape, Add, UpSampling2D 


class NHDRRNet(Model):
    def __init__(self, config):
        super(NHDRRNet, self).__init__()

        self.attention_filter = config.attention_filter
        self.filter = config.filter
        self.encoder_kernel = config.encoder_kernel
        self.decoder_kernel = config.decoder_kernel
        self.triple_pass_filter = config.triple_pass_filter

    def adaptive_interpolation(self, required_size, img):
        pool_size = (int(required_size[0]/img.shape[1]), int(required_size[1]/img.shape[2]))
        return UpSampling2D(size=pool_size)(img) 

    def attention_network(self, I_l, I_h):
        concat = tf.concat([I_l, I_h], axis=-1)
        lay1 = Conv2D(self.attention_filter, self.encoder_kernel, padding='same', activation='relu')(concat)
        out = Conv2D(6, self.encoder_kernel, padding='same', activation='sigmoid')(lay1)
        return out

    def encoder_1(self, X, i):
        X = Conv2D(int(self.filter*i), self.encoder_kernel, strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        return X

    def decoder_last(self, X):
        X = Conv2DTranspose(3, self.decoder_kernel, strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)
        return X

    def decoder(self, X, i):
        X = Conv2DTranspose(int(self.filter*i), self.decoder_kernel, strides=(2,2), padding='same')(X)
        X = BatchNormalization()(X)
        X = LeakyReLU()(X)
        return X

    def triplepass(self, T0):
        T1 = Conv2D(self.triple_pass_filter, kernel_size=(1,1), strides=(1,1), padding='same')(T0)
        T1 = ReLU()(T1)

        T2 = Conv2D(self.triple_pass_filter, kernel_size=(3,3), strides=(1,1), padding='same')(T0)
        T2 = ReLU()(T2)

        T3 = Conv2D(self.triple_pass_filter, kernel_size=(5,5), strides=(1,1), padding='same')(T0)
        T3 = ReLU()(T3)

        T3 = Add()([T1, T2, T3])

        T4 = Conv2D(self.triple_pass_filter, kernel_size=(3,3), strides=(1,1), padding='same')(T3)
        T5 = Add()([T4, T0])

        return T5

    def global_non_local(self, X):
        h, w , c = list(X.shape)[1], list(X.shape)[2], list(X.shape)[3]
        theta = Conv2D(128, kernel_size=(1,1), padding='same')(X)
        theta_rsh = Reshape((h*w, 128))(theta)

        phi = Conv2D(128, kernel_size=(1,1), padding='same')(X)
        phi_rsh = Reshape((128, h*w))(phi)

        g = Conv2D(128, kernel_size=(1,1), padding='same')(X)
        g_rsh = Reshape((h*w, 128))(phi)

        theta_phi = tf.matmul(theta_rsh, phi_rsh)
        theta_phi = tf.keras.layers.Softmax()(theta_phi)

        theta_phi_g = tf.matmul(theta_phi, g_rsh)
        theta_phi_g = Reshape((h, w, 128))(theta_phi_g)

        theta_phi_g = Conv2D(256, kernel_size=(1,1), padding='same')(theta_phi_g)

        out = Add()([theta_phi_g, X])

        return out

    def main_model(self, X):
        ## attention network
        X_1 = X[:,0,:,:,:]
        X_2 = X[:,1,:,:,:]
        X_3 = X[:,2,:,:,:]

        mask1 = self.attention_network(X_1, X_2)
        mask2 = self.attention_network(X_3, X_2)

        X_1_masked = tf.math.multiply(mask1, X_1)
        X_3_mask = tf.math.multiply(mask2, X_2)

        X_concat = tf.concat([X_1_masked, X_2, X_3_mask], axis=-1)
        X_concat = Conv2D(64, kernel_size=(1,1), padding='same')(X_concat)

        X_32 = self.encoder_1(X_concat, 1)
        X_64 = self.encoder_1(X_32, 2)
        X_128 = self.encoder_1(X_64, 4)
        X_256 = self.encoder_1(X_128, 8)

        ## upper path ##
        tpl_out = self.triplepass(X_256)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)
        tpl_out = self.triplepass(tpl_out)

        ## lower path ##
        adt_layer = AdaptiveAveragePooling2D(output_size=(16, 16))(X_256)
        glb_local = self.global_non_local(adt_layer)
        glb_local = self.adaptive_interpolation(required_size=list(tpl_out.shape)[1:3], img=glb_local)

        ## cat ##
        concat = tf.concat([tpl_out, glb_local], axis=-1)
        O_128 = self.decoder(tpl_out, 4)
        O_128 = Add()([X_128, O_128])

        O_64 = self.decoder(O_128, 2)
        O_64 = Add()([X_64, O_64])
        O_32 = self.decoder(O_64, 1)
        O_32 = Add()([X_32, O_32])

        out = self.decoder_last(O_32)

        return out



