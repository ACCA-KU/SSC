import D4CMPP
# get the current directory
import os
import sys
import inspect


def train(**kwargs):
    if kwargs.get('network',None) not in ['SGC', 'SGC_GCN', 'SGC_MPNN', 'SGC_DMPNN', 'SGC_AFP', 'SGCwoPE', 'SGCwoPE_GCN', 'SGCwoPE_MPNN', 'SGCwoPE_DMPNN', 'SGCwoPE_AFP']:
        raise ValueError("network must be one of ['SGC', 'SGC_GCN', 'SGC_MPNN', 'SGC_DMPNN', 'SGC_AFP', 'SGCwoPE', 'SGCwoPE_GCN', 'SGCwoPE_MPNN', 'SGCwoPE_DMPNN', 'SGCwoPE_AFP']")

    pwd  = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    kwargs['NET_REFER'] = os.path.join(pwd, "network_refer.yaml")
    kwargs['NET_DIR'] = os.path.join(pwd, "SGCnet")
    kwargs['DataManager_PATH'] = os.path.join(pwd, "SGCDataManager")
    kwargs['explicit_h_columns'] = ['solvent']
    kwargs['sculptor_index'] = (6,2,0)

    return D4CMPP.train(**kwargs)
