from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


def fc_net():
    l = [   make_connected_layer(3072, 324),
            make_activation_layer(RELU),
            make_connected_layer(324, 288),
            make_activation_layer(RELU),
            make_connected_layer(288, 64),
            make_activation_layer(RELU),
            make_connected_layer(64, 16),
            make_activation_layer(RELU),
            make_connected_layer(16, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = fc_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How many operations does the convnet use during a forward pass?
#   32 * 32 * 3 * 8 * 3 * 3
# + 16 * 16 * 8 * 16 * 3 * 3
# + 8 * 8 * 16 * 32 * 3 * 3
# + 4 * 4 * 32 * 64 * 3 * 3
# + 256 * 10
# = 1108480 (operations)

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# conv_net has a training accuracy of 0.708 and a test accuracy of 0.662;
# fc_net has a training accuracy of 0.423 and a test accuracy of 0.422.
# conv_net performs better because it can understand spatial relations between pixels better.
# and conv_net share weight to find patterns in the images no matther where the patterns locate.
