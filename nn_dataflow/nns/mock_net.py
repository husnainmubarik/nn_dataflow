""" $lic$
Copyright (C) 2016-2020 by Tsinghua University and The Board of Trustees of
Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer

'''
AlexNet

Krizhevsky, Sutskever, and Hinton, 2012
'''

#def __init__(self, nifm, nofm, sofm, sfil, strd=1):

NN = Network('MockNet')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1_a', ConvLayer(3, 48, 55, 11, 4), prevs=(NN.INPUT_LAYER_KEY,))
#NN.add('conv1_b', ConvLayer(3, 48, 55, 11, 4), prevs=(NN.INPUT_LAYER_KEY,))
#NN.add('pool1_a', PoolingLayer(48, 27, 3, strd=2), prevs=('conv1_a',))
#NN.add('pool1_b', PoolingLayer(48, 27, 3, strd=2), prevs=('conv1_b',))
# Norm layer is ignored.

#NN.add('conv2_a', ConvLayer(48, 128, 27, 5), prevs=('pool1_a',))
#NN.add('conv2_b', ConvLayer(48, 128, 27, 5), prevs=('pool1_b',))
#NN.add('pool2_a', PoolingLayer(128, 13, 3, strd=2), prevs=('conv2_a',))
#NN.add('pool2_b', PoolingLayer(128, 13, 3, strd=2), prevs=('conv2_b',))
