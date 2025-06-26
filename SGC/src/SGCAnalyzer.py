
import numpy as np
import torch
import io
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
from D4CMPP import Analyzer

ISAAnalyzer = Analyzer.ISAAnalyzer_v1p3

class SGCAnalyzer(ISAAnalyzer):
        
    def check_score_by_group(self):
        if getattr(self.dm.gg,'sculptor',None) is None:
            self.is_score_by_group = True
        else:
            temp_smiles='CC(C)(C)OC(=O)C1=CC=CC=C1C(=O)O'
            inputs = {}
            for c in self.molecule_columns:
                inputs[c] = [temp_smiles]
            for c in self.numeric_input_columns:
                inputs[c] = [0.0]
            
            test_loader,_ = self.prepare_temp_data(inputs)
            temp_score = self.tm.get_score(self.nm, test_loader)

            self.is_score_by_group = True

