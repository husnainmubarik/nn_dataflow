import sys
import math

# nn_dataflow imports
from .layer import Layer, ConvLayer, LocalRegionLayer
from .phy_dim2 import PhyDim2
from .. import util
from . import mem_hier_enum as me
from . import data_category_enum as de


class ValueCost():
    def __init__(self, lbs, layer, cost, batch_size):
      self.lbs = lbs
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
      #TODO: Fix/recheck/verify the loop dimensions as per Eyeriss mapping
      local_cost = 0
      ops = self.lbs.ops
      dram_fil_accesses = self.lbs.nld.total_access_at_of(me.DRAM, de.FIL)
      wfil = self.layer.wfil
      weights_accessed = dram_fil_accesses * wfil
      counter = 0
      for i in range(math.ceil(ops / weights_accessed)):
        for j in range(int(weights_accessed)):
            try:
              wei = self.cost.my_weights['conv1'][j]
            except IndexError:
              print('Error while reading weights')
              wei = 100
              #sys.exit()
            if wei < 0:
              key = 'multn_'+str(-1*wei)
            else:
              key = 'multp_'+str(wei)
            local_cost += self.cost.value_mult[key]
            counter += 1
            #print(local_cost)
      print('ops counter: {}'.format(counter))
      print('Results: value specific mult cost is: {}'.format(local_cost))
      return local_cost

    def value_adder_cost(self):
      #TODO: Fix/recheck/verify the loop dimensions as per Eyeriss mapping
      local_cost = 0
      ops = self.lbs.ops
      dram_fil_accesses = self.lbs.nld.total_access_at_of(me.DRAM, de.FIL)
      wfil = self.layer.wfil
      weights_accessed = dram_fil_accesses * wfil
      counter = 0
      for i in range(math.ceil(ops / weights_accessed)):
        for j in range(int(weights_accessed)):
            try:
              wei = self.cost.my_weights['conv1'][j]
            except IndexError:
              print('Error while reading weights')
              wei = 100
              #sys.exit()
            if wei < 0:
              key = 'multn_'+str(-1*wei)
            else:
              key = 'multp_'+str(wei)
            local_cost += self.cost.value_mult[key]
            counter += 1
            #print(local_cost)
      print('ops counter: {}'.format(counter))
      print('Results: value specific mult cost is: {}'.format(local_cost))
      return local_cost

    def value_control_logic_cost(self):
      local_cost = 0
      #TODO: Fix the loop dimensions as per Eyeriss logical mapping
      dram_ifm_accesses = self.lbs.nld.total_access_at_of(me.DRAM, de.IFM)
      dram_fil_accesses = self.lbs.nld.total_access_at_of(me.DRAM, de.FIL)
      wfil = self.layer.wfil
      weights_accessed = dram_fil_accesses * wfil
      #key = 'multp_0'
      #counter = 0
      #for i in range(math.ceil(dram_ifm_accesses / weights_accessed)):
      #  for j in range(int(weights_accessed)):
      #    try:
      #      wei = self.cost.my_weights['conv1'][j]
      #    except IndexError:
      #      wei = 100
      #      print('Error while reading weights')
      #    if wei < 0:
      #      key = 'multn_'+str(-1*wei)
      #    else:
      #      key = 'multp_'+str(wei)
      #    local_cost += self.cost.value_control[key]
      #    counter += 1
      #print('control counter: {}'.format(counter))
      local_cost = self.cost.value_control * dram_ifm_accesses
      print('Results: value specific control logic cost is: {}'.format(local_cost))
      return local_cost

    def value_control_regf_cost(self):
      local_cost = 0
      #TODO: weight Regfile is accessed for each ifmap, recheck this assumption 
      dram_ifm_accesses = self.lbs.nld.total_access_at_of(me.DRAM, de.IFM)
      local_cost = dram_ifm_accesses * self.cost.mem_hier_at(me.REGF) 
      print('Results: value specific control regf cost is: {}'.format(local_cost))
      return local_cost

    def value_gbuf_cost(self):
      local_cost = 0
      #TODO: weight Regfile is accessed for each ifmap, recheck this accumption 
      #dram_ifm_accesses = self.lbs.nld.total_access_at_of(me.DRAM, de.IFM)
      access = self.lbs.remote_gbuf_access
      cos = self.cost.mem_hier_at(me.GBUF) 
      local_cost = 0
      print('i access: {}'.format(access))
      print('i cos: {}'.format(cos))
      print('Results: value specific control gbuf cost is: {}'.format(local_cost))
      return local_cost

    def value_adder_cost(self):
      local_cost = 0
      #TODO: Effectively each operation is added to each other?  

      ops = self.lbs.ops

      local_cost = (ops / 2) * self.cost.adder_cost
      print('Results: value specific adder cost is: {}'.format(local_cost))
      return local_cost

    def value_pes_regf_cost(self):
      local_cost = 0
      #TODO: Fix the loop dimensions as per Eyeriss logical mapping
      dram_accesses = self.layer.wfil * self.layer.hfil * self.cnt_lpeset 
      key = 'multp_0'
      for i in range(int(dram_accesses)):
        try:
          wei = self.cost.my_weights['conv1'][i]
        except IndexError:
          wei = 100
          print('Error while reading weights')
        if wei < 0:
          key = 'multn_'+str(-1*wei)
        else:
          key = 'multp_'+str(wei)
        #print(key)
        local_cost += self.cost.value_control[key]
      return local_cost

    def value_dram_access_cost(self):
      local_cost = 0
      #TODO: Fix the loop dimensions as per Eyeriss logical mapping
      dram_accesses = self.layer.wfil * self.layer.hfil * self.cnt_lpeset 
      key = 'multp_0'
      for i in range(int(dram_accesses)):
        try:
          wei = self.cost.my_weights['conv1'][i]
        except IndexError:
          wei = 100
          print('Error while reading weights')
        if wei < 0:
          key = 'multn_'+str(-1*wei)
        else:
          key = 'multp_'+str(wei)
        #print(key)
        local_cost += self.cost.value_control[key]
      return local_cost

    def value_gbuf_access_cost(self):
      local_cost = 0
      #TODO: Fix the loop dimensions as per Eyeriss logical mapping
      dram_accesses = self.layer.wfil * self.layer.hfil * self.cnt_lpeset 
      key = 'multp_0'
      for i in range(int(dram_accesses)):
        try:
          wei = self.cost.my_weights['conv1'][i]
        except IndexError:
          wei = 100
          print('Error while reading weights')
        if wei < 0:
          key = 'multn_'+str(-1*wei)
        else:
          key = 'multp_'+str(wei)
        #print(key)
        local_cost += self.cost.value_control[key]
      return local_cost

    def value_noc_cost(self):
      local_cost = 0
      #TODO: Fix the loop dimensions as per Eyeriss logical mapping
      dram_accesses = self.layer.wfil * self.layer.hfil * self.cnt_lpeset 
      key = 'multp_0'
      for i in range(int(dram_accesses)):
        try:
          wei = self.cost.my_weights['conv1'][i]
        except IndexError:
          wei = 100
          print('Error while reading weights')
        if wei < 0:
          key = 'multn_'+str(-1*wei)
        else:
          key = 'multp_'+str(wei)
        #print(key)
        local_cost += self.cost.value_control[key]
      return local_cost
