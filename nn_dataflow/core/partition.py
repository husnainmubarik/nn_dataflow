""" $lic$
Copyright (C) 2016-2017 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

If you use this program in your research, we request that you reference the
TETRIS paper ("TETRIS: Scalable and Efficient Neural Network Acceleration with
3D Memory", in ASPLOS'17. April, 2017), and that you send us a citation of your
work.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

import itertools
import fastcache

from . import data_category_enum as de
from . import parallel_enum as pe
from .. import util
from .fmap_range import FmapPosition, FmapRange
from .int_range import IntRange
from .layer import ConvLayer, LocalRegionLayer
from .partition_scheme import PartitionScheme
from .phy_dim2 import PhyDim2

'''
Parallel process partitioning.

Partition among multiple copies of the PE arrays.

For our case, only deal with up to 2D layout of PE arrays.
'''

def gen_partition(layer, batch_size, dim_nodes, options, guaranteed=False):
    '''
    Generator for all possible partitioning schemes that partition `layer` into
    2D `dim_nodes` nodes.

    If `guaranteed` is True, we guarantee to yield at least one partitioning
    scheme regardless of efficiency.
    '''
    # pylint: disable=too-many-branches

    yielded = False

    for ph, pw in itertools.product(util.factorize(dim_nodes.h, pe.NUM),
                                    util.factorize(dim_nodes.w, pe.NUM)):

        pdims = [PhyDim2(h, w) for h, w in zip(ph, pw)]

        # Batch partitoning.
        if (not options.partition_batch) and pdims[pe.BATP].size() > 1:
            continue
        elif batch_size % pdims[pe.BATP].size() != 0:
            continue

        if options.partition_hybrid:
            # Require partition is approximately dividable of total size.
            if not util.approx_dividable(layer.nofm, pdims[pe.OUTP].size()):
                continue
            if not util.approx_dividable(layer.hofm, pdims[pe.OFMP].h) \
                    or not util.approx_dividable(layer.wofm, pdims[pe.OFMP].w):
                continue

            if options.partition_dimensional:
                # Force each partitioning to be only along one dimension,
                # except if there is only one partitioning.
                if any(pd.h != 1 and pd.w != 1 for pd in pdims) \
                        and sum(pd.size() > 1 for pd in pdims) > 1:
                    continue

            if (not options.partition_ifmaps) and pdims[pe.INPP].size() > 1:
                continue
            else:
                if isinstance(layer, ConvLayer):
                    if not util.approx_dividable(layer.nifm,
                                                 pdims[pe.INPP].size()):
                        continue
                elif isinstance(layer, LocalRegionLayer):
                    if pdims[pe.INPP].size() > 1:
                        continue
        else:
            assert not options.partition_ifmaps
            if pdims[pe.INPP].size() != 1:
                continue

            if layer.hofm == 1 and layer.wofm == 1:
                # FC layer: no OFMP.
                if pdims[pe.OFMP].size() != 1:
                    continue
            else:
                # CONV layer: no OUTP.
                if pdims[pe.OUTP].size() != 1:
                    continue

        # Skip the transpose equivalence. Only partitioning scheme without OFMP
        # and with only 1-D partitioning could have equivalence, since we
        # always index in height-major order.
        if pdims[pe.OFMP].size() == 1 \
                and all(pd.h == 1 or pd.w == 1 for pd in pdims):
            pdhs, pdws = zip(*pdims)
            if pdhs > pdws:
                continue

        # For different order.
        for order in itertools.permutations(range(pe.NUM)):

            # Partition with dim-1 has no effect, so its order is not relevant.
            skip = False
            for idx in range(pe.NUM - 1):
                pae1 = order[idx]
                pae2 = order[idx + 1]
                pdim1 = pdims[pae1]
                pdim2 = pdims[pae2]

                # Invalid cases include:
                # - both are (1, 1) but not in order of ParallelEnum.
                # - (1, 1) after non-(1, 1).
                # - (1, non-1) after (non-1, 1) of not BATP.
                if pdim1.size() == 1 and pdim2.size() == 1 and pae1 > pae2:
                    skip = True
                    break
                if pdim1.size() > 1 and pdim2.size() == 1:
                    skip = True
                    break
                if pae1 != pe.BATP and pdim2.h == 1 and pdim2.w > 1 \
                        and pdim1.h > 1 and pdim1.w == 1:
                    skip = True
                    break
            if skip:
                continue

            no_part = [pae for pae in range(pe.NUM) if pdims[pae].size() == 1]
            # Batch parallelism should be at the top.
            if pe.BATP not in no_part and order[len(no_part)] != pe.BATP:
                continue

            part = PartitionScheme(order, pdims)
            assert part.dim() == dim_nodes

            yield part

            yielded = True

    if guaranteed and not yielded:
        # None of the Partitioning schemes are valid. May be due to
        # non-dividability. Return a single naive scheme, with only OFMP or
        # only OUTP.

        pdims = [PhyDim2(1, 1)] * pe.NUM
        order = range(pe.NUM)

        if layer.hofm == 1 and layer.wofm == 1:
            # Only OUTP, no OFMP.
            pdims[pe.OUTP] = dim_nodes
        else:
            # Only OFMP, no OUTP.
            pdims[pe.OFMP] = dim_nodes

        part = PartitionScheme(order, pdims)
        assert part.dim() == dim_nodes

        yield part


def proc_data_range(layer, batch_size, part, pidx):
    '''
    Get the partitioned data ranges of the batched layer, including filter
    range, ifmap range, and ofmap range, for the given processing node with
    partition index `pidx`.

    Filter range is returned as a tuple of ((i_beg, i_end), (o_beg, o_end));
    i/ofmap ranges are returned as FmapRange instances.
    '''

    # Partitioned ofmap range.

    ofrng = part.fmap_range(FmapRange((0,) * 4,
                                      FmapPosition(b=batch_size, n=layer.nofm,
                                                   h=layer.hofm, w=layer.wofm)),
                            pidx)

    # Partitioned ifmap range.

    # Derived from the partitioned ofmap range.
    b_orng, n_orng, h_orng, w_orng = ofrng.beg_end('b', 'n', 'h', 'w')

    # Batch partition.
    b_beg, b_end = b_orng

    if isinstance(layer, ConvLayer):
        # Ifmap channel partition.
        idx_ifm = pidx[pe.INPP].h * part.dim(pe.INPP).w + pidx[pe.INPP].w
        n_beg, n_end = util.get_ith_range((0, layer.nifm),
                                          idx_ifm, part.size(pe.INPP))
        # Fmap height tiling.
        h_beg, h_end = h_orng
        # xy_i = xy_o * stride + (0 ... sfil-1)
        h_beg = h_beg * layer.htrd
        h_end = max(h_beg, (h_end - 1) * layer.htrd + layer.hfil)

        # Fmap width tiling.
        w_beg, w_end = w_orng
        w_beg = w_beg * layer.wtrd
        w_end = max(w_beg, (w_end - 1) * layer.wtrd + layer.wfil)

    elif isinstance(layer, LocalRegionLayer):
        # Ifmap channel partition.
        n_beg, n_end = n_orng
        n_beg = max(0, n_beg - layer.nreg // 2)
        n_end = min(layer.nifm, n_end + layer.nreg - layer.nreg // 2)

        # Fmap height tiling.
        h_beg, h_end = h_orng
        h_beg = h_beg * layer.htrd
        h_end = max(h_beg, (h_end - 1) * layer.htrd + layer.hreg)

        # Fmap width tiling.
        w_beg, w_end = w_orng
        w_beg = w_beg * layer.wtrd
        w_end = max(w_beg, (w_end - 1) * layer.wtrd + layer.wreg)

    assert n_end <= layer.nifm and h_end <= layer.hifm and w_end <= layer.wifm

    ifrng = FmapRange(FmapPosition(b=b_beg, n=n_beg, h=h_beg, w=w_beg),
                      FmapPosition(b=b_end, n=n_end, h=h_end, w=w_end))

    # Filter range.

    if isinstance(layer, ConvLayer):
        filrng = (ifrng.beg_end('n'), ofrng.beg_end('n'))
    elif isinstance(layer, LocalRegionLayer):
        # No filter.
        filrng = (IntRange(0, 0), IntRange(0, 0))

    return filrng, ifrng, ofrng


@fastcache.clru_cache(maxsize=1024)
def unit_nhops_to_proc_region(layer, batch_size, region, part,
                              filter_nodes, ifmap_layout, ofmap_layout,
                              options):
    '''
    Get the total number of hops to transfer each data category to the
    processing node region once. The layer computation is partitioned with
    PartitionScheme `part` on NodeRegion `region`.

    Since the filters are read-only and independent of the previous layer
    computation, we duplicate the filters in multiple nodes given by
    `filter_nodes`, and assume the accesses can be forwarded to the nearest
    node. `ifmap_layout` and `ofmap_layout` specify the data layouts of the
    i/ofmaps as `DataLayout` instances.

    Return a tuple with each element being the number of hops for each data
    category.
    '''

    del options

    # FmapRange --> list of node coordinates processing this data.
    fil_dict = {}
    ofm_dict = {}
    ifm_dict = {}

    for pidx in part.gen_pidx():
        coord = part.coordinate(region, pidx)
        filrng, ifrng, ofrng = proc_data_range(layer, batch_size, part, pidx)

        if not filrng[0].empty() and not filrng[1].empty():
            fil_dict.setdefault(filrng, []).append(coord)
        if ifrng.size() > 0:
            ifm_dict.setdefault(ifrng, []).append(coord)
        if ofrng.size() > 0:
            ofm_dict.setdefault(ofrng, []).append(coord)

    # All data should be processed by the same number of nodes, or no node.
    assert len(set(len(v) for v in fil_dict.values())) <= 1, \
            'fil val len: {}'.format([len(v) for v in fil_dict.values()])
    assert len(set(len(v) for v in ifm_dict.values())) <= 1, \
            'ifm val len: {}'.format([len(v) for v in ifm_dict.values()])
    assert len(set(len(v) for v in ofm_dict.values())) <= 1, \
            'ofm val len: {}'.format([len(v) for v in ofm_dict.values()])

    fil_dict = util.HashableDict.fromdict(fil_dict, valfunc=tuple)
    ifm_dict = util.HashableDict.fromdict(ifm_dict, valfunc=tuple)
    ofm_dict = util.HashableDict.fromdict(ofm_dict, valfunc=tuple)

    nhops = [0] * de.NUM

    nhops[de.FIL] = _unit_nhops_to_fil(layer, filter_nodes, fil_dict)

    nhops[de.IFM] = _unit_nhops_to_ifm(ifmap_layout, ifm_dict)

    if ofmap_layout.parts == (part,) and ofmap_layout.regions == (region,):
        # Ofmaps are stored locally, no data transfer.
        pass
    else:
        nhops[de.OFM] = _unit_nhops_to_ofm(ofmap_layout, ofm_dict)

    return nhops


@fastcache.clru_cache(maxsize=1024)
def _unit_nhops_to_fil(layer, filter_nodes, fil_dict):
    '''
    Get the total number of hops to transfer filter data.

    `fil_dict` maps each filter range to the nodes that process it.
    '''
    nhops = 0

    for filrng, coord_list in fil_dict.items():
        fil_size = filrng[0].size() * filrng[1].size() * layer.filter_size()

        # Min hops to each processing node across all filter source nodes.
        min_hops = [min(coord.hop_dist(c) for c in filter_nodes)
                    for coord in coord_list]
        nhops += fil_size * sum(min_hops)

    return nhops


@fastcache.clru_cache(maxsize=1024)
def _unit_nhops_to_ifm(ifmap_layout, ifm_dict):
    '''
    Get the total number of hops to transfer ifmap data.

    `ifm_dict` maps each fmap range to the nodes that process it.
    '''
    nhops = 0

    for ifrng, coord_list in ifm_dict.items():
        nhops += ifmap_layout.nhops_to(ifrng, *coord_list)

    return nhops


@fastcache.clru_cache(maxsize=1024)
def _unit_nhops_to_ofm(ofmap_layout, ofm_dict):
    '''
    Get the total number of hops to transfer ofmap data.

    `ofm_dict` maps each fmap range to the nodes that process it.
    '''
    nhops = 0

    for ofrng, coord_list in ofm_dict.items():
        # Additional synchronization is necessary between INPP nodes. Only one
        # node fetches the previously-partial-accumulated data from memory into
        # its buffer and start on it. Other nodes start on zero and send the
        # results to that node to accumulate there.

        # Use the mid node.
        mid_idx = len(coord_list) // 2
        for idx, coord in enumerate(coord_list):
            if idx == mid_idx:
                # The mid node. Fetch from memory.
                nhops += ofmap_layout.nhops_to(ofrng, coord)
            else:
                # Others. Send to the mid node (one way).
                dist = coord.hop_dist(coord_list[mid_idx])
                nhops += util.idivc(ofrng.size() * dist, 2)

    return nhops

