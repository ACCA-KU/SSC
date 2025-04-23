import numpy as np

from D4CMPP.src.DataManager.ISADataManager import ISADataManager
from D4CMPP.src.utils import PATH

from .ISAGraphDataset_withSolv import ISAGraphDataset_withSolv

class ISADataManager_withSolv(ISADataManager):
    def __init__(self, config):
        super(ISADataManager_withSolv, self).__init__(config)
        self.molecule_columns.append('solvent')
        self.molecule_graphs={col:[] for col in self.molecule_columns}

    def import_others(self):
        super(ISADataManager_withSolv, self).import_others()
        self.dataset = ISAGraphDataset_withSolv
        self.unwrapper = self.dataset.unwrapper

    def init_temp_data(self,smiles_list,solvents_list):
        self.molecule_smiles['compound'] = np.array(smiles_list)
        self.valid_smiles['compound'] = np.array(smiles_list)
        self.molecule_smiles['solvent'] = np.array(solvents_list)
        self.valid_smiles['solvent'] = np.array(solvents_list)
        self.molecule_graphs={col:[] for col in self.molecule_columns}
        self.set = None
        self.target_value = np.zeros((len(smiles_list),self.config["target_dim"]))
        self.gg.verbose= False
            
        for col in self.molecule_columns:
            self.molecule_smiles[col] = np.array(smiles_list) if col == 'compound' else np.array(solvents_list)
            self.generate_graph(col)

        self.prepare_dataset(temp=True)
        return self.valid_smiles

    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.molecule_graphs['solvent'], self.target_value, self.molecule_smiles['compound'], self.molecule_smiles['solvent'])
    