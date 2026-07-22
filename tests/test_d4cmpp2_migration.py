"""Regression tests for the D4CMPP2/PyG SSC migration."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import torch
from torch_geometric.data import Batch, HeteroData

from D4CMPP2.networks.base import MolecularNetwork
from D4CMPP2.networks.registry import get_model


NETWORK_MODULES = (
    "SSC",
    "SSC_GCN",
    "SSC_MPNN",
    "SSC_DMPNN",
    "SSC_AFP",
    "SSConlyPE",
    "SSCwoPE_GCN",
    "SSCwoPE_MPNN",
    "SSCwoPE_DMPNN",
    "SSCwoPE_AFP",
)


def _isa_graph(node_dim=8, edge_dim=5):
    graph = HeteroData()
    graph["r_nd"].x = torch.randn(3, node_dim)
    graph["i_nd"].x = torch.ones(3, 1)
    graph["d_nd"].x = torch.ones(2, 1)

    real_edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph["r_nd", "r2r", "r_nd"].edge_index = real_edges
    graph["r_nd", "r2r", "r_nd"].edge_attr = torch.randn(4, edge_dim)
    graph["i_nd", "i2i", "i_nd"].edge_index = real_edges.clone()
    graph["i_nd", "i2i", "i_nd"].edge_attr = torch.ones(4, 1)
    graph["i_nd", "i2d", "d_nd"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 1]]
    )
    graph["d_nd", "d2d", "d_nd"].edge_index = torch.tensor(
        [[0, 1], [1, 0]]
    )
    graph["d_nd", "d2d", "d_nd"].edge_attr = torch.randn(2, 4)
    graph["d_nd", "d2r", "r_nd"].edge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 2]]
    )
    return graph


def _batch():
    compound = Batch.from_data_list([_isa_graph(), _isa_graph()])
    solvent = Batch.from_data_list([_isa_graph(), _isa_graph()])
    return {
        "compound_graphs": compound,
        "compound_r_node": compound["r_nd"].x,
        "compound_i_node": compound["i_nd"].x,
        "compound_r2r_edge": compound["r_nd", "r2r", "r_nd"].edge_attr,
        "compound_d2d_edge": compound["d_nd", "d2d", "d_nd"].edge_attr,
        "solvent_graphs": solvent,
        "solvent_r_node": solvent["r_nd"].x,
        "solvent_i_node": solvent["i_nd"].x,
        "solvent_r2r_edge": solvent["r_nd", "r2r", "r_nd"].edge_attr,
        "solvent_d2d_edge": solvent["d_nd", "d2d", "d_nd"].edge_attr,
    }


class MigrationTests(unittest.TestCase):
    def test_networks_follow_the_registered_molecular_network_contract(self):
        import SSC

        for module_name in NETWORK_MODULES:
            with self.subTest(network=module_name):
                module = importlib.import_module(f"SSC.src.SSCnet.{module_name}")
                model_class = module.network
                self.assertTrue(issubclass(model_class, MolecularNetwork))
                self.assertEqual(model_class.model_name, module_name)
                definition = get_model(module_name)
                self.assertIs(definition.network, model_class)
                self.assertEqual(definition.data_contract, "isa")
                training_config = definition.training_config()
                self.assertEqual(training_config["version"], "2.0")
                self.assertEqual(
                    training_config["data_manager_class"], "ISADataManager"
                )
                self.assertEqual(
                    set(model_class.optimization_space()),
                    {
                        "hidden_dim",
                        "conv_layers",
                        "linear_layers",
                        "dropout",
                        "solvent_dim",
                        "solvent_conv_layers",
                    },
                )
        self.assertIs(get_model("SSCwoPE-2"), get_model("SSCwoPE_GCN"))
        self.assertIn("SSC", SSC.VALID_NETWORKS)
        self.assertNotIn("SSC_multi", SSC.VALID_NETWORKS)
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            get_model("SSC_multi")

    def test_model_config_and_input_validation_use_d4cmpp2_contract(self):
        from SSC.src.SSCnet.SSC import SSC

        caller_config = {"node_dim": 8, "edge_dim": 5, "target_dim": 1}
        model = SSC(caller_config)
        self.assertEqual(caller_config, {"node_dim": 8, "edge_dim": 5, "target_dim": 1})
        self.assertEqual(model.config["solvent_dim"], 64)
        with self.assertRaisesRegex(ValueError, "compound_graphs"):
            model()

    def test_all_standard_networks_forward_backward_and_score(self):
        config = {
            "node_dim": 8,
            "edge_dim": 5,
            "target_dim": 1,
            "hidden_dim": 8,
            "solvent_dim": 8,
            "conv_layers": 2,
            "solvent_conv_layers": 2,
            "linear_layers": 2,
            "dropout": 0.0,
        }
        for module_name in NETWORK_MODULES:
            with self.subTest(network=module_name):
                module = importlib.import_module(f"SSC.src.SSCnet.{module_name}")
                model = module.network(dict(config))
                batch = _batch()
                prediction = model(**batch)
                self.assertEqual(tuple(prediction.shape), (2, 1))
                loss = model.compute_loss(prediction, torch.zeros_like(prediction))
                loss.backward()
                score = model(**_batch(), get_score=True)
                expected_score_keys = (
                    {"RP", "PEF"}
                    if module_name == "SSConlyPE"
                    else {"RP", "SC", "PEF"}
                )
                self.assertEqual(set(score), expected_score_keys)

    def test_custom_data_manager_is_d4cmpp2_pyg_adapter(self):
        from D4CMPP2.src.DataManager.ISADataManager import ISADataManager
        from SSC.src.SSCDataManager.ISADataManager_withSolv import (
            ISADataManager_withSolv,
        )

        self.assertTrue(issubclass(ISADataManager_withSolv, ISADataManager))

    @unittest.skipUnless(
        os.environ.get("SSC_RUN_INTEGRATION") == "1",
        "set SSC_RUN_INTEGRATION=1 for public train-save-reload-Analyzer smoke",
    )
    def test_public_train_save_reload_analyzer(self):
        from SSC import Analyzer, train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "tiny_ssc.csv"
            (root / "models").mkdir()
            (root / "graphs").mkdir()
            pd.DataFrame(
                {
                    "compound": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF"],
                    "solvent": ["O", "CO", "CCO", "O", "CO", "CCO"],
                    "value": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                    "set": ["train", "train", "train", "train", "val", "test"],
                }
            ).to_csv(data_path, index=False)
            model_path = train(
                data=str(data_path),
                target=["value"],
                network="SSC",
                device="cpu",
                max_epoch=1,
                batch_size=2,
                MODEL_DIR=str(root / "models"),
                GRAPH_DIR=str(root / "graphs"),
                data_quality_report=False,
                random_seed=7,
            )
            self.assertTrue((Path(model_path) / "final.pth").is_file())
            saved_config = (Path(model_path) / "config.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("network_id: SSC", saved_config)
            self.assertIn("version: '2.0'", saved_config)
            self.assertNotIn("DataManager_PATH", saved_config)
            self.assertNotIn("NET_DIR", saved_config)
            self.assertTrue((Path(model_path) / "network_identity.json").is_file())
            saved_network = (Path(model_path) / "network.py").read_text(
                encoding="utf-8"
            )
            self.assertIn("class SSC(SSCMolecularNetwork)", saved_network)
            analyzer = Analyzer(model_path, device="cpu")
            prediction = analyzer.predict(["CCO"], ["O"])
            self.assertIn(("CCO", "O"), prediction)
            score = analyzer.get_score("CCO", "O")
            self.assertEqual(set(score), {"RP", "SC", "PEF"})

    @unittest.skipUnless(
        os.environ.get("SSC_RUN_PACKAGING") == "1",
        "set SSC_RUN_PACKAGING=1 for wheel-only import smoke",
    )
    def test_wheel_contains_runtime_modules_and_network_reference(self):
        project_root = Path(__file__).resolve().parents[1]
        d4cmpp2_package = Path(os.environ["D4CMPP2_SOURCE_ROOT"]).resolve()
        d4cmpp2_import_root = d4cmpp2_package.parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel_dir = root / "dist"
            install_dir = root / "install"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_dir),
                    str(project_root),
                ],
                check=True,
            )
            wheel = next(wheel_dir.glob("SSC-*.whl"))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(install_dir),
                    str(wheel),
                ],
                check=True,
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(install_dir), str(d4cmpp2_import_root)]
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import SSC; "
                        "import SSC.src.SSCnet.SSC; "
                        "from D4CMPP2.networks.registry import get_model; "
                        "assert get_model('SSC').network.model_name == 'SSC'; "
                        "p=Path(SSC.__file__).parent/'src/network_refer.yaml'; "
                        "assert p.is_file(), p"
                    ),
                ],
                cwd=root,
                env=environment,
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
