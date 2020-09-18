import tensorflow as tf 
import numpy as np

from tensorflow.keras.applications import vgg19

width, height = 256, 256
img_nrows = 200
img_ncols = int(width * img_nrows / height)

total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input((img*255.0).astype(np.uint8))
    return tf.convert_to_tensor((img/255.0).astype(np.float32))

# def gram_matrix(x):
#     x = tf.transpose(x, (2, 0, 1))
#     features = tf.reshape(x, (tf.shape(x)[0], -1))
#     gram = tf.matmul(features, tf.transpose(features))
#     return gram

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def style_loss(gt, output):
    S = gram_matrix(gt)
    C = gram_matrix(output)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(gt, output):
    return tf.reduce_sum(tf.square(output - gt))


# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in
# VGG19 (as a dict).
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=outputs_dict)


# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# The layer to use for the content loss.
content_layer_name = "block5_conv2"


def compute_loss(gt, output):
    input_tensor = tf.concat([gt, output], axis=0)
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    gt_features = layer_features[:16, :, :, :]
    output_features = layer_features[16:, :, :, :]
    loss = loss + content_weight * content_loss(
        gt, output
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        gt_features = layer_features[:16, :, :, :]
        output_features = layer_features[16:, :, :, :]
        sl = style_loss(gt_features, output_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    # loss += total_variation_weight * total_variation_loss(output)
    return loss