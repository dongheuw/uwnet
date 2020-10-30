#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // DONE: 6.1 - iterate over the input and fill in the output with max values
    for(int r = 0; r < in.rows; ++r) {
        int c = r * out.cols;
        for (int ch = 0; ch < l.channels; ++ch) {
            for (int row = 0; row < l.height; row += l.stride) {
                for (int col = 0; col < l.width; col += l.stride) {
                    float val_max = -1e18;
                    for (int i = -(l.size - 1) / 2; i < -(l.size - 1) / 2 + l.size; ++i) {
                        for (int j = -(l.size - 1) / 2; j < -(l.size - 1) / 2 + l.size; ++j) {
                            if (row + i >= 0 && row + i < l.height && col + j >= 0 && col + j < l.width) {
                                float candidate = in.data[ch * l.height * l.width + r * in.cols + (row + i) * l.width + col + j];
                                if (candidate > val_max)
                                    val_max = candidate;
                            }

                        }
                    }
                    out.data[c++] = val_max;
                }
            }
        }
    }
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    // DONE: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    for(int r = 0; r < in.rows; ++r) {
        int c = r * dy.cols;
        for (int ch = 0; ch < l.channels; ++ch) {
            for (int row = 0; row < l.height; row += l.stride) {
                for (int col = 0; col < l.width; col += l.stride) {
                    float val_max = -1e18;
                    int val_max_pos = -1;
                    for (int i = -(l.size - 1) / 2; i < -(l.size - 1) / 2 + l.size; ++i) {
                        for (int j = -(l.size - 1) / 2; j < -(l.size - 1) / 2 + l.size; ++j) {
                            if (row + i >= 0 && row + i < l.height && col + j >= 0 && col + j < l.width) {
                                float candidate = in.data[ch * l.height * l.width + r * in.cols + (row + i) * l.width + col + j];
                                if (candidate > val_max) {
                                    val_max = candidate;
                                    val_max_pos = l.width * row + col + i * l.width + j + ch * l.height * l.width + r * in.cols;
                                }
                            }

                        }
                    }
                    dx.data[val_max_pos] += dy.data[c++];
                }
            }
        }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

