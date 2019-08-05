import numpy as np
import cv2

class convolution:

    def __init__(self,fanin,fanout):
        self.bias=np.zeros((1,1,1,64))
        self.filter=np.random.randn(3,3,fanin,fanout)*np.sqrt(2/fanin)*0.01

    def conv_forward(self,inputimage, filter, stride, pad, bias):

        (batchsize, input_height, input_width, input_chan) = inputimage.shape
        (n_height, n_width, n_channels, n_filters) = filter.shape

        future_height = ((input_height - n_height + 2 * pad) / stride) + 1
        future_width = ((input_width - n_width + 2 * pad) / stride) + 1

        future_image = np.zeros((batchsize, int(future_height), int(future_width), n_filters))

        inputimage = np.pad(inputimage, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

        for i in range(batchsize):
            inputimage = inputimage[i]
            for k in range(input_height):
                for m in range(input_width):
                    for c in range(n_filters):
                        h_init = k * stride
                        h_end = h_init + 3
                        w_init = m * stride
                        w_end = w_init + 3
                        inputpart = inputimage[h_init:h_end, w_init:w_end, :]
                        future_image[i, k, m, c] = self.convinput(inputpart, filter[:, :, :, c], bias[:, :, :, c])
        cache = (inputimage, filter, stride, pad, bias)
        return future_image, cache

    def convinput(self,input, filter, bias):
        t = filter * input + bias
        return np.sum(t)
    def conv_backward(self,dZ, cache):




        (inputimage, filter, stride, pad, bias) = cache


        (batchsize, inp_height, inp_width, inp_chan) = inputimage.shape


        (n_height, n_width, n_channels, n_filters) = filter.shape


        (batchsize, height, width, chan) = dZ.shape


        dinputimage = np.zeros((batchsize, inp_height, inp_width, inp_chan))
        dWeight = np.zeros((f, f, inp_chan, chan))
        dbias = np.zeros((1, 1, 1, chan))


        inputimage = np.pad(inputimage, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        dinputimage = np.pad(dinputimage, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

        for i in range(batchsize):


            inputimage = inputimage[i]
            dinputimage = dinputimage[i]

            for h in range(height):
                for w in range(width):
                    for c in range(chan):


                        vert_start = h * stride

                        vert_end = vert_start + n_height
                        horiz_start = w * stride

                        horiz_end = horiz_start + n_width


                        a_slice = inputimage[vert_start:vert_end, horiz_start:horiz_end, :]


                        dinputimage[vert_start:vert_end, horiz_start:horiz_end, :] += filter[:, :, :, c] * dZ[
                            i, h, w, c]
                        dWeight[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        dbias[:, :, :, c] += dZ[i, h, w, c]


            dinputimage[i, :, :, :] = dinputimage[pad:-pad, pad:-pad, :]



        assert (dinputimage.shape == (batchsize, inp_height, inp_width, inp_chan))

        return dinputimage, dWeight, dbias