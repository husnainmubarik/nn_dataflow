import unittest
import sys
import pdb

sys.path.append('/home/mubarik3/DASICS/mult_exploration/mult_synthesis')
sys.path.append('../')

from nn_dataflow.core import Cost
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer
from nn_dataflow.core import MapStrategy, MapStrategyEyeriss
from nn_dataflow.core import MemHierEnum as me
from nn_dataflow.core import Network
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import NNDataflow
from nn_dataflow.core import Option
from nn_dataflow.core import PhyDim2
from nn_dataflow.core import Resource
from nn_dataflow.nns import import_network
from nn_dataflow.tools.nn_dataflow_search import *

# value aware imports 
from read_json import readValueMult8Cost

class TestNNDataflow(unittest.TestCase):
    ''' Tests for NNDataflow module. '''

    def __init__(self):

        self.alex_net = import_network('alex_net')
        
        self.map_strategy = MapStrategyEyeriss
        
        value_mult = {}
        self.cost = Cost(value_mult = value_mult,
                         mac_op=1,
                         mem_hier=(200, 6, 2, 1),
                         noc_hop=0,
                         idl_unit=0)

        self.options = Option()
        

    def eyerissAsplos17(self):
        '''
        Reproduce TETRIS ASPLOS'17 paper Figure 8.
        '''
        network = self.alex_net

        batch_size = 16


        resource = Resource(proc_region=NodeRegion(origin=PhyDim2(0, 0),
                                                   dim=PhyDim2(4, 4),
                                                   type=NodeRegion.PROC),
                            dram_region=NodeRegion(
                                origin=PhyDim2(0, 0), dim=PhyDim2(4, 4),
                                type=NodeRegion.DRAM),
                            src_data_region=NodeRegion(
                                origin=PhyDim2(0, 0), dim=PhyDim2(4, 4),
                                type=NodeRegion.DRAM),
                            dst_data_region=NodeRegion(
                                origin=PhyDim2(0, 0), dim=PhyDim2(4, 4),
                                type=NodeRegion.DRAM),
                            dim_array=PhyDim2(14, 14),
                            size_gbuf=133032 // 2,  # 133 kB
                            size_regf=512 // 2,  # 512 B
                            array_bus_width=float('inf'),
                            dram_bandwidth=float('inf'),
                            no_time_mux=False,
                           )

        mult_cost = readValueMult8Cost()
        print(mult_cost['multp_5'])

        cost = Cost(value_mult=mult_cost,
                    mac_op=2e-12,
                    mem_hier=(80e-12, 14e-12, 4e-12, 0.6e-12),  # pJ/16-b
                    noc_hop=40e-12,
                    idl_unit=200e-12)

        options = Option(sw_gbuf_bypass=(True, True, True),
                         sw_solve_loopblocking=True,
                         partition_hybrid=True)
        
        #pdb.set_trace()

        nnd = NNDataflow(network, batch_size, resource, cost,
                         self.map_strategy)
        tops, _ = nnd.schedule_search(options)
        self.assertTrue(tops)
        dfsch_t16 = tops[0]

        ## Check results.

        # Same workload.
        #self.assertAlmostEqual(dfsch_t16.total_ops, dfsch_l1.total_ops)
        print('t16 ops: {}'.format(dfsch_t16.total_ops))

        # Performance of T-16 is proportional to PE resource (20% margin).
        #self.assertLess(dfsch_t16.total_time,
        #                1.2 * dfsch_l1.total_time * (16 * 16) / (14 * 14 * 16))
        print('t16_time: {}'.format(dfsch_t16.total_time))
        
        # Energy reduced by > 30%.
        # self.assertLess(dfsch_t16.total_cost, dfsch_l1.total_cost * 0.7)
        # With dimension restriction on partitioning, this is slightly violated.
        #self.assertLess(dfsch_t16.total_cost, dfsch_l1.total_cost * 0.72)
        print('t16_energy: {}'.format(dfsch_t16.total_cost))
        for i in dfsch_t16:
          print(str(i)+',')
        ## Check results.

        # Results as cost for each component:
        header = 'ALU, DRAM, Buffer, Array, RF'
        cost_bkdn = {}

        for layer in dfsch_t16:
            layer = str(layer)
            op_cost = 0
            access_cost = [0] * me.NUM

            for layer_part in network:
                if not layer_part or not layer_part.startswith(layer):
                    continue
                sr = dfsch_t16[layer_part]
                op_cost += sr.total_ops * cost.mac_op
                access_cost = [ac + a * c for ac, a, c
                               in zip(access_cost, sr.total_accesses,
                                      cost.mem_hier)]

            cost_bkdn[layer] = []
            # To 1e9.
            cost_bkdn[layer].append(op_cost *1e12 / 1e9)
            cost_bkdn[layer].append(access_cost[me.DRAM] *1e12 / 1e9)
            cost_bkdn[layer].append(access_cost[me.GBUF] *1e12 / 1e9)
            cost_bkdn[layer].append(access_cost[me.ITCN] *1e12 / 1e9)
            cost_bkdn[layer].append(access_cost[me.REGF] *1e12 / 1e9)

        for layer in cost_bkdn:
            print(cost_bkdn[layer])


if __name__ == '__main__':
  print('ran')
  obj = TestNNDataflow()
  obj.eyerissAsplos17()
