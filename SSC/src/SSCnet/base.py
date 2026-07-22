"""D4CMPP2 network contracts shared by all SSC model variants."""

from types import MappingProxyType

from D4CMPP2.networks.base import Hyperparameter, InputContract, MolecularNetwork


SSC_HYPERPARAMETERS = MappingProxyType(
    {
        "hidden_dim": Hyperparameter(
            "int", default=64, low=1, search_low=16, search_high=256,
            step=16, grid=(32, 64, 128, 256),
            description="Hidden width of the compound ISA branch.",
        ),
        "conv_layers": Hyperparameter(
            "int", default=4, low=1, search_low=1, search_high=8,
            step=1, grid=(2, 4, 6, 8),
            description="Number of compound message-passing layers.",
        ),
        "linear_layers": Hyperparameter(
            "int", default=3, low=1, search_low=1, search_high=5,
            step=1, grid=(1, 2, 3, 4),
            description="Historical SSC prediction-head depth setting.",
        ),
        "dropout": Hyperparameter(
            "float", default=0.1, low=0.0, high=0.5,
            search_low=0.0, search_high=0.5,
            grid=(0.0, 0.1, 0.2, 0.3, 0.5),
            description="Dropout probability.",
        ),
        "solvent_dim": Hyperparameter(
            "int", default=64, low=1, search_low=16, search_high=256,
            step=16, grid=(32, 64, 128),
            description="Hidden width of the solvent branch.",
        ),
        "solvent_conv_layers": Hyperparameter(
            "int", default=4, low=1, search_low=1, search_high=8,
            step=1, grid=(2, 4, 6),
            description="Number of solvent message-passing layers.",
        ),
    }
)

SSC_OPTIMIZATION_SPACE = tuple(SSC_HYPERPARAMETERS)

SSC_INPUT_CONTRACT = InputContract(
    required=(
        "compound_graphs",
        "compound_r_node",
        "compound_i_node",
        "compound_r2r_edge",
        "compound_d2d_edge",
        "solvent_graphs",
        "solvent_r_node",
        "solvent_r2r_edge",
    ),
    optional=("target", "get_score"),
)

class SSCMolecularNetwork(MolecularNetwork):
    """Common D4CMPP2 contract for compound/solvent SSC networks."""

    required_config = ("node_dim", "edge_dim", "target_dim")
    input_contract = SSC_INPUT_CONTRACT
    hyperparameters = SSC_HYPERPARAMETERS
    default_optimization_space = SSC_OPTIMIZATION_SPACE
    config_aliases = MappingProxyType(
        {
            "solvent_hidden_dim": "solvent_dim",
            "solv_hidden_dim": "solvent_dim",
            "solv_conv_layers": "solvent_conv_layers",
        }
    )

    def forward(self, **batch):
        raise NotImplementedError


__all__ = [
    "SSC_HYPERPARAMETERS",
    "SSC_INPUT_CONTRACT",
    "SSC_OPTIMIZATION_SPACE",
    "SSCMolecularNetwork",
]
