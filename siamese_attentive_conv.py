from keras import Input
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation, dot, Lambda
from attention_layer import AttentionLayer
import keras.backend as K

# from keras examples
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


input =  Input(shape=(None,config.embeddings_size), dtype='float32')

conv_4 = Conv1D(config.nr_filters*3,
                4,
                padding='same',
                activation='relu',
                strides=1)(input)
shared = Model(input, conv_4)

pr_1 =  Input(shape=(None, 300), dtype='float32')
pr_2 =  Input(shape=(None, 300), dtype='float32')

out_1 = shared(pr_1)
out_2 = shared(pr_2)

attention = AttentionLayer()([out_1,out_2])

# out_1 column wise
att_1 = GlobalMaxPooling1D()(attention)
att_1 = Activation('softmax')(att_1)
out_1 = dot([att_1, out_1], axes=1)

# out_2 row wise
attention_transposed = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(attention)
att_2 = GlobalMaxPooling1D()(attention_transposed)
att_2 = Activation('softmax')(att_2)
out_2 = dot([att_2, out_2], axes=1)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([out_1, out_2])

model = Model(input=[pr_1, pr_2], output=distance)
