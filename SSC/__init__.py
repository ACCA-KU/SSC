import D4CMPP
# get the current directory
import os
import sys
import inspect

from .src.SSCAnalyzer import SSCAnalyzer as Analyzer
import yaml
def train(**kwargs):
    """
    Train the network for SSC with the given configuration.

    Args for the training:
        (Required args for the scratch training)
        data        : str
                      the name or path of the data file as a csv file. you can omit the ".csv" extension.
        target      : list[str]
                      the name of the target column.
        network     : str. default= "SSC"
                      the name of the network to use. 
                      ['SSC', 'SSC_GCN', 'SSC_MPNN', 'SSC_DMPNN', 'SSC_AFP', 'SSCwoPE', 'SSCwoPE_GCN', 'SSCwoPE_MPNN', 'SSCwoPE_DMPNN', 'SSCwoPE_AFP']
    
        (Network hyperparameters. The following args will be varied according to the network chosen.)
        hidden_dim      : int. Default= 64
                        the dimension of the hidden layers in the graph convolutional layers
        solvent_dim    : int. Default= 64
                        the dimension of the hidden layers in the solvent graph convolutional layers
        conv_layers     : int. Default= 6
                        the number of graph convolutional layers
        solvent_conv_layers: int. Default= 4
                        the number of graph convolutional layers for the solvent
        linear_layers   : int. Default= 3
                        the number of linear layers after the graph convolutional layers
        dropout         : float. Default= 0.1

        (Reqired args for continuing the training)
        LOAD_PATH   : str
                      the path of the directory that contains the model to continue the training.

        (Required args for the transfer learning)
        TRANSFER_PATH: str
                      the path of the directory that contains the model to transfer the learning.
        data        : str
        target      : list[str]
        lr_dict     : dict (optional)
                      the dictionary for the learning rate of specific layers. e.g. {'GCNs': 0.0001}
                      you can find the layer names in "model_summary.txt" in the model directory.

                                    
        (Optional args)
        explicit_h_columns: list[str]. Default= []
                        the name of the columns to be used as explicit hydrogen features for the nodes.
        scaler      : str. Default= "standard",
                      ['standard', 'minmax', 'normalizer', 'robust', 'identity']
        optimizer   : str. Default= "Adam",
                      ['Adam', 'SGD', 'RMSprop', ... supported by torch.optim]
        max_epoch   : int. Defualt= 2000,
        batch_size  : int. Default= 128,
        learning_rate: float. Defualt= 0.001,
        weight_decay: float. Default= 0.0005,
        lr_patience : int. Defualt= 80,
                      the number of epochs with no improvement after which learning rate will be reduced.
        early_stopping_patience: 200,
                      the number of epochs with no improvement after which training will be stopped
        min_lr      : float. Defult= 1e-5,
                      the minimum learning rate
        device      : str. Defualt='cuda:0'
                      ['cpu', 'cuda:0', 'cuda:1', ...]
        pin_memory  : bool. Default= False
                      If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        split_random_seed: int. Default= 42,
            
        DATA_PATH   : str. Default= None
                      the path of directory that contains the data file. If None, it will walk the subpath.
        NET_REFER   : str. Default= "{src}/network_refer.yaml"
                      the path to the network reference file. 
        MODEL_DIR   : str. Default= "./_Models"
                      the path to the directory to save the models.
        GRAPH_DIR  : str. Default= "./_Graphs"
                      the path to the directory to save the graphs.
        FRAG_REF    : str. Default= "{src}/utils/functional_group.csv"
                      the path to the reference file for the functional groups.


    --------------------------------------------
    Example:

        train(data='data.csv', target=['target1','target2'], network='GCN',)

        train(LOAD_PATH='GCN_model_test_Abs_20240101_000000')

        train(TRANSFER_PATH='GCN_model_test_Abs_20240101_000000', data='data.csv', target=['target1','target2'], lr_dict={'GCNs': 0.0001})

        train(data='data.csv', target=['target1','target2'], network='GCN', hidden_dim=32, conv_layers=4, linear_layers=2, dropout=0.1)

    
    """
    # Check if the required arguments are provided
    valid_networks = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "src/network_refer.yaml"), 'r')).keys()
    if kwargs.get('network','SSC') not in valid_networks:
        raise ValueError(f"network must be one of {valid_networks}")
    if kwargs.get('network',None) is None:
        kwargs['network'] = 'SSC'
    pwd  = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    kwargs['NET_REFER'] = os.path.join(pwd, "src/network_refer.yaml")
    kwargs['NET_DIR'] = os.path.join(pwd, "src/SSCnet")
    kwargs['DataManager_PATH'] = os.path.join(pwd, "src/SSCDataManager")
    kwargs['explicit_h_columns'] = ['solvent']
    kwargs['sculptor_index'] = (6,2,0)

    return D4CMPP.train(**kwargs)
