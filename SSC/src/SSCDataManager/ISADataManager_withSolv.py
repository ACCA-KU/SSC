"""D4CMPP2 adapter for SSC's ISA compound/solvent input pair."""

from D4CMPP2.src.DataManager.ISADataManager import ISADataManager


class ISADataManager_withSolv(ISADataManager):
    def __init__(self, config):
        columns = list(config.get("molecule_columns", ["compound"]))
        if "compound" not in columns:
            columns.insert(0, "compound")
        if "solvent" not in columns:
            columns.append("solvent")
        config["molecule_columns"] = columns
        super().__init__(config)
