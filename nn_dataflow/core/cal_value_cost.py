import sys

# nn_dataflow imports
from .layer import Layer, ConvLayer, LocalRegionLayer
from .phy_dim2 import PhyDim2
from .. import util

#def value_logic_cost(ops, wfil, wofm, hfil, hofm, 
#                     ops_lpe,
#                     size_dim_lpeset,
#                     cnt_lpeset
#                     ):
#  print(''' value cost func ops: {}, wfil: {}, wofm: {}, hfil: {}, hofm: {},
#           ops_lpe: {}, size_dim_lpeset: {}, cnt_lpeset: {}'''
#           .format(ops, wfil, wofm, hfil, hofm, ops_lpe, 
#                   size_dim_lpeset, cnt_lpeset
#                   ))  
class ValueCost():
  def __init__(self, layer, cost, batch_size):
    self.layer = layer
    self.cost = cost
    self.batch_size = batch_size
    

    # Calculations taken from nn_dataflow
    if isinstance(self.layer, ConvLayer):
        # Conv and FC layers.
        self.ops_lpe = self.layer.wfil * self.layer.wofm
        self.dim_lpeset = PhyDim2(self.layer.hfil, self.layer.hofm)
        self.cnt_lpeset = self.batch_size * self.layer.nofm * self.layer.nifm
    
    self.ops_logic_total = self.ops_lpe * self.dim_lpeset.size() \
                           * self.cnt_lpeset
  
  def value_logic_cost(self):
    print('Hey testing: layer is {}'.format(self.layer))
    local_cost = 0
    d2_filter = self.layer.wfil * self.layer.hfil
    counter = 0
    #TODO: Fix the loop dimensions as per Eyeriss logical mapping
    for cnt in range(self.cnt_lpeset):
      for hf in range(self.layer.wofm):
        for wf in range(self.layer.wfil):
          try:
            wei = self.cost.my_weights['conv1'][wf+counter]
          except IndexError:
            print('Error while reading weights')
            wei = 100
            #sys.exit()
        
          if wei < 0:
            key = 'multn_'+str(-1*wei)
          else:
            key = 'multp_'+str(wei)
          #print(key)
          local_cost += self.cost.value_mult[key]
      counter += self.layer.hfil * self.layer.wfil
    print('Hey testing value mult cost is: {}'.format(local_cost))
    return local_cost
  #
  #def value_control_cost(self, dram_accesses):
  #  #print('Hey testing len 
  #  #        of weight is {}'.format(len(self.cost.my_weights['conv1'])))
  #  #print('Hey testing ops are {}'.format(lbs_ops))
  #  #sys.exit()
  #  local_cost = 0
  #  for i in range(int(dram_accesses)):
  #    try:
  #      wei = self.cost.my_weights['conv1'][i]
  #    except IndexError:
  #      wei = 100      #TODO: Some mock number (num_weights<ops)will fix 
  #                     #index error later 
  #                     #Clue: need to check lbs_ops calculations
  #    if wei < 0:
  #      key = 'multn_'+str(-1*wei)
  #    else:
  #      key = 'multp_'+str(wei)
  #    #print(key)
  #    local_cost += self.cost.value_control[key]
  #  return local_cost
