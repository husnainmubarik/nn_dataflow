import sys
import math

# nn_dataflow imports
from .layer import Layer, ConvLayer, FCLayer, PoolingLayer, LocalRegionLayer
from .phy_dim2 import PhyDim2
from .. import util
from . import mem_hier_enum as me
from . import data_category_enum as de

#TODO: need to fix batch sizes everywhere 
layer_names = {'conv1_a':'conv1',
                'conv1_b':'conv1',
                'conv2_a':'conv2',
                'conv2_b':'conv2',
                'conv3_a':'conv3',
                'conv3_b':'conv3',
                'conv4_a':'conv4',
                'conv4_b':'conv4',
                'conv5_a':'conv5',
                'conv5_b':'conv5',
                'fc1':'fc6',
                'fc2':'fc7',
                'fc3':'fc8',
                'pool1_a':'pool1_a',
                'pool1_b':'pool1_b',
                'pool2_a':'pool2_a',
                'pool2_b':'pool2_b',
                'pool3_a':'pool3_a',
                'pool3_b':'pool3_b'

                }
class ValueCost():
    def __init__(self, layer, cost, resource, batch_size, layer_name):
      self.layer = layer
      self.cost = cost
      self.batch_size = batch_size
      self.resource = resource
      self.layer_name = layer_names[layer_name]

      #self.layer_name = 'conv1' #TODO: fix layer names corresponding to the 
                                #layer name in the weight file and the actual
                                #layer name in the network 
                                #hint: trivial manage dict of layer names

      # Calculations taken from nn_dataflow
      if isinstance(self.layer, ConvLayer):
          # Conv and FC layers.
          pass

    def value_logic_cost(self):
      #TODO: Fix/recheck/verify the loop dimensions as per Eyeriss mapping
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        ops = self.layer.total_ops()
        total_weights = self.layer.total_filter_size()
        counter = 0
        for i in range(math.ceil(ops / total_weights)):
          for j in range(int(total_weights)):
              try:
                wei = self.cost.my_weights[self.layer_name][j]
              except IndexError as e:
                print('''Error while reading weights in value logic cost: {}'''
                          .format(e))
                wei = 100 # mock value will be captured in the output
              if wei < 0:
                key = 'multn_'+str(-1*wei)
              else:
                key = 'multp_'+str(wei)
              local_cost += self.cost.value_mult[key]
              counter += 1
              #print(local_cost)

        assert counter == ops, \
                ('''Value logic cost calculated cost is for ops {}, while actual 
                    ops are {}'''.format(counter,ops))

        print('Results: value specific PE-matrix cost is: {}'.format(local_cost))
        return local_cost

    def value_logic_timing_for_all_ofm(self):
      '''
      # compare each weight with previous weight to add two cycles
      # if previous weight value was same as current weight value
      # as PE would be busy calculating last weight, as we want to
      # keep track of prev_weight as long as we\'re calculating one ofm
      # to stay consistent with nndflow
      '''
      #TODO: Fix/recheck/verify the loop dimensions as per Eyeriss mapping
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        ops = self.layer.total_ops()
        filter_size = self.layer.filter_size()
        total_weights = self.layer.total_filter_size()
        ops_counter = 0
        num_pes = self.resource.num_value_pes
        weights_for_one_ofm_cal = total_weights // self.layer.nofm

        for i in range(ops//total_weights):
          weight_counter = 0
          prev_wei = 0
          num_busy_pes = 0
          for j in range(total_weights//weights_for_one_ofm_cal):
            for k in range(weights_for_one_ofm_cal):
              try:
                wei = self.cost.my_weights[self.layer_name][weight_counter]
              except IndexError as e:
                wei = 100 # Mock value, error will be captured in output
                print('''Error while reading weights in logic timing
                       for all ofm calculations: {}'''.format(e))

              if wei == prev_wei or num_busy_pes >= num_pes:
                local_cost += 1
                num_busy_pes = 0

              prev_wei = wei
              weight_counter += 1
              ops_counter += 1
              num_busy_pes += 1

            local_cost += 1

        assert ops_counter == ops, \
                ('''calculated cost is for ops {}, while actual ops are {}'''
                    .format(ops_counter,ops))
        print('''Results: value specific PE-matrix cycles are: {}'''
                  .format(local_cost))
        return local_cost

    def value_logic_timing_for_one_ofm(self):
      '''
      # compare each weight with previous weight to add two cuycles
      # if previous weight value was same as current weight value
      # as PE would be busy calculating last weight, as we want to
      # keep track of prev_weight as long as we\'re calculating one ofm
      # to stay consistent with nndflow
      '''
      #TODO: Fix/recheck/verify the loop dimensions as per Eyeriss mapping
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        ifmaps = self.layer.ifmap_size()
        filter_size = self.layer.filter_size()
        total_weights = self.layer.total_filter_size()
        ops_counter = 0
        num_pes = self.resource.num_value_pes
        weights_for_one_ofm_cal = total_weights // self.layer.nofm

        #print('ifmaps: {}, filtersize: {}, weights: {}, w for one ofm: {}'
        #        .format(ifmaps,filter_size,total_weights,weights_for_one_ofm_cal))

        prev_wei = 0
        num_busy_pes = 0
        for j in range(ifmaps//weights_for_one_ofm_cal):
          weight_counter = 0
          for k in range(weights_for_one_ofm_cal):
            try:
              wei = self.cost.my_weights[self.layer_name][weight_counter]
            except IndexError as e:
              wei = 100 # Mock value, error will be captured in output
              print('''Error while reading weights in logic timing
                     for one ofm calculations: {}'''.format(e))

            if wei == prev_wei or num_busy_pes >= num_pes:
              local_cost += 1
              num_busy_pes = 0

            prev_wei = wei
            weight_counter += 1
            ops_counter += 1
            num_busy_pes += 1

          local_cost += 1

        #TODO: add appropriate assertion to chech correctness
        #assert ops_counter == ops, \
        #        ('''calculated cost is for ops {}, while actual ops are {}'''
        #            .format(ops_counter,ops))
        print('''Results: value specific PE-matrix cycles for one ofm are: {}'''
                  .format(local_cost))
        return local_cost

    def value_memory_access_timing_for_one_ofm(self):
      '''
      # we will calculate dram access time based on the dram accesses 
      # considering that we want to compute one ofm at a time which means
      # we will load filters as well as 
      '''
      #TODO: Fix/recheck/verify the loop dimensions as per Eyeriss mapping
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        ops = self.layer.total_ops()
        filter_size = self.layer.filter_size()
        total_weights = self.layer.total_filter_size()
        ops_counter = 0
        num_pes = self.resource.num_value_pes
        weights_for_one_ofm_cal = total_weights // self.layer.nofm
        weights_load_time = weights_for_one_ofm_cal \
                            * self.cost.mem_cycles[me.GBUF]
        ifm_load_time = self.layer.ifmap_size() \
                            * self.cost.mem_cycles[me.GBUF]
        local_cost = weights_load_time + ifm_load_time
        print('''Results: value specific memory cycles for one ofm: {}'''
              .format(local_cost))
        return local_cost

    def value_memory_access_timing_for_all_ofm(self):
      '''
      '''
      #TODO: Fix/recheck/verify the loop dimensions as per Eyeriss mapping
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        ops = self.layer.total_ops()
        filter_size = self.layer.filter_size()
        total_weights = self.layer.total_filter_size()
        ops_counter = 0
        num_pes = self.resource.num_value_pes
        weights_for_all_ofm_cal = total_weights
        weights_load_time = weights_for_all_ofm_cal \
                            * self.cost.mem_cycles[me.GBUF]
        ifm_load_time = self.layer.total_ifmap_size() \
                            * self.cost.mem_cycles[me.GBUF]
        local_cost = weights_load_time + ifm_load_time
        print('''Results: value specific memory cycles for all ofm: {}'''
              .format(local_cost))
        return local_cost


    def value_control_logic_cost(self):
      #TODO: Fix the loop dimensions as per Eyeriss logical mapping
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        total_ifms = self.layer.total_ifmap_size(batch_size = self.batch_size)
        local_cost = self.cost.value_control * total_ifms
        print('''Results: value specific control logic cost is: {}'''
                .format(local_cost))
        return local_cost

    def value_control_regf_cost(self):
      '''
      # control regf stores weights and is accessed to route ifmaps to pes
      # it should be accessed ifmaps times then
      '''
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        total_ifms = self.layer.total_ifmap_size(batch_size = self.batch_size)
        local_cost = total_ifms * self.cost.mem_hier_at(me.REGF)
        print('''Results: value specific control regf cost is: {}'''
                  .format(local_cost))
        return local_cost

    def value_gbuf_cost(self):
      '''
      # gbuf is accessed for all weights, ifmaps and ofmaps
      '''
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        total_weights = self.layer.total_filter_size()
        total_ifms = self.layer.total_ifmap_size(batch_size = self.batch_size)
        total_ofms = self.layer.total_ofmap_size(batch_size = self.batch_size)
        accesses = total_weights + total_ifms + total_ofms
        local_cost = accesses * self.cost.mem_hier_at(me.GBUF)
        print('''Results: value specific gbuf access cost is: {}'''
                  .format(local_cost))
        return local_cost

    def value_dram_cost(self):
      '''
      # dram is accessed for all weights ifmaps and ofmaps
      '''
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        total_weights = self.layer.total_filter_size()
        total_ifms = self.layer.total_ifmap_size(batch_size = self.batch_size)
        total_ofms = self.layer.total_ofmap_size(batch_size = self.batch_size)
        accesses = total_weights + total_ifms + total_ofms
        local_cost = accesses * self.cost.mem_hier_at(me.DRAM)
        print('''Results: value specific dram access cost is: {}'''
                  .format(local_cost))
        return local_cost

    def value_adder_cost(self):
      '''
      # Effectively each operation would to go through addition?
      '''
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        ops = self.layer.total_ops()
        local_cost = (ops) * self.cost.adder_cost
        print('Results: value specific adder cost is: {}'.format(local_cost))
        return local_cost

    def value_pes_regf_cost(self):
      '''
      # refg will be accessed once for each operation to read IFM only
      '''
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        ops = self.layer.total_ops()
        local_cost = ops * self.cost.mem_hier_at(me.REGF)
        return local_cost

    def value_noc_cost(self):
      '''
      # Each op needs to be first sent to pes then result to adder
      '''
      #TODO: Come up with some assertions to check correctness
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        ops = self.layer.total_ops()
        hops = 2 * ops
        local_cost = hops * self.cost.noc_hop
        print('''Results: value specific noc cost is: {}'''
                  .format(local_cost))
        return local_cost

    def value_static_cost(self, time):
      '''
      '''
      #TODO: Come up with some assertions to check correctness
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = time * self.cost.idl_unit
        print('''Results: value specific static cost is: {}'''
                  .format(local_cost))
        return local_cost
    
    def proposed_arch_cycles(self):
      '''
      # compare each weight with previous weight to add extra cycle
      # if previous weight value was same as current weight value
      # as PE would be busy calculating last weight, as we want to
      # keep track of prev_weight as long as we\'re calculating one ofm
      # to stay consistent with nndflow
      '''
      #TODO: Fix/recheck/verify the loop dimensions as per Eyeriss mapping
      if isinstance(self.layer, ConvLayer) or isinstance(self.layer, FCLayer):
        local_cost = 0
        ops = self.layer.total_ops()
        filter_size = self.layer.filter_size()
        total_weights = self.layer.total_filter_size()
        ops_counter = 0
        #num_pes = self.resource.num_value_pes
        num_pes = 256
        num_clusters = 100
        num_units = num_clusters*num_pes
        weights_for_one_ofm_cal = total_weights // self.layer.nofm

        print('weights for one ofm are: {}'.format(weights_for_one_ofm_cal))
        counter = 0

        # 25600
        # clusters 100
        # pe per cluster 256

        for i in range(25600//256):
          prev_wei = 'nan'
          for j in range(256):
            wei = self.cost.my_weights[self.layer_name][counter]
            counter += 1
            if wei == prev_wei:
              local_cost += 1
            prev_wei = wei

        #for i in range(ops//total_weights):
        #  weight_counter = 0
        #  for j in range(total_weights//num_units):
        #    for k in range(num_units // num_pes):
        #      prev_wei = 0
        #      if weight_counter > 25600:
        #        break
        #      for l in range(num_pes):
        #        try:
        #          wei = self.cost.my_weights[self.layer_name][weight_counter]
        #        except IndexError as e:
        #          wei = 100 # Mock value, error will be captured in output
        #          print('''Error while reading weights in logic timing
        #                 for all ofm calculations: {}'''.format(e))

        #        if wei == prev_wei:
        #          local_cost += 1

        #        prev_wei = wei
        #        ops_counter += 1
        #        weight_counter += 1

        #  local_cost += 1

        #assert ops_counter == ops, \
        #        ('''calculated cost is for ops {}, while actual ops are {}'''
        #            .format(ops_counter,ops))
        ops_counter = 25600
        print('''Results: value specific PE-matrix cycles are: {} ops are {}'''
                  .format(local_cost,ops_counter))
        exit()
        return local_cost
