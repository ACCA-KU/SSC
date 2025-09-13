
import numpy as np
import torch
import io
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
from D4CMPP import Analyzer

ISAAnalyzer = Analyzer.ISAAnalyzer_v1p3

class SSCAnalyzer(Analyzer.ISAwSAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_score_by_group = True
        
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

    # get the attention score of the given smiles
    def get_score(self, smiles, solvent, inverse_transform=False, key = None, is_key_relative=False):
        "get the attention score of the given smiles by its subgroups"
        test_loader,_ = self.prepare_temp_data([smiles],[solvent])
        result = self.tm.get_score(self.nm, test_loader)
        for k in result.keys():
            if type(result[k]) is torch.Tensor:
                result[k] = result[k].detach().cpu().numpy()


        if key is None:
            if inverse_transform:
                result = self.inverse_transform(result)
            return result
        elif key not in result:
            raise ValueError(f"Key '{key}' not found in the result. Available keys: {list(result.keys())}")
        else:
            result = result[key]
            if inverse_transform:
                result = self.inverse_transform(result, is_relative=is_key_relative)
            return self.get_group_score(smiles, result)
        


    def plot_score(self, smiles, solvent ,key, inverse_transform=False, **kwargs):
        """
        This function plots the attention score of the given smiles by its subgroups.

        Args:
            smiles (str): The smiles of the molecule.
            atom_with_index (bool): Whether to show the atom index or not.
            score_scaler (function): The function to scale the score. Default is lambda x:x.
            ticks (list): The ticks of the colorbar. Default is [0,0.25,0.5,0.75,1].
            rot (int): The rotation of the molecule. Default is 0.
            locate (str): The location of the subplots. Default is 'right'. It can be 'right' or 'bottom'.
            figsize (float): The size of the figure. Default is 1.
            only_total (bool): Whether to plot only the total score or plot PAS and NAS as well. Default is False.
            with_colorbar (bool): Whether to show the colorbar or not. Default is True.

        Returns:
            dict: The attention score of the given smiles.
                  Each value in the dictionary is the attention score of corresponding atom with the same index.
        
        """
        score = self.get_score(smiles, solvent)
        score['positive'] = score[key]
        result = self._plot_score(smiles, score, **kwargs)
        if inverse_transform:
            result= self.inverse_transform(result)
        return result
    

        
    def plot_subgroup_score_histogram(self, smiles_list, solvent_list, nums = 10, bins=40, xlim=[0,1],inverse_transform=False):
        """
        This function plots the histogram of the attention score of the given list of smiles by its subgroups.

        Args:
            smiles_list (list): The list of smiles.
            nums (int): The number of subgroups to plot. Default is 10.
                        The most frequent subgroups will be plotted.
            bins (int): The number of bins. Default is 40.
            xlim (list): The range of the x-axis. Default is [0,1].
        
        Returns:
            dict: PAS of subgroups 
            dict: NAS of subgroups 
        
        """
        pos_frag, neg_frag = self.get_subgroup_score_bin(smiles_list, solvent_list, inverse_transform=inverse_transform)
        
        pos_frag = dict(sorted(pos_frag.items(), key=lambda x: len(x[1]), reverse=True)[:nums])
        for i,f in enumerate(pos_frag):
            plt.hist(pos_frag[f], bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=0.5, label=f)
        plt.xlim(xlim[0], xlim[1])
        plt.legend()
        plt.title('Positive')
        plt.show()

        if len(neg_frag) == 0:
            return pos_frag, neg_frag
        
        neg_frag = dict(sorted(neg_frag.items(), key=lambda x: len(x[1]), reverse=True)[:nums])
        for i,f in enumerate(neg_frag):
            plt.hist(neg_frag[f], bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=0.5, label=f)
        plt.xlim(xlim[0], xlim[1])
        plt.legend()
        plt.title('Negative')
        plt.show()
        return pos_frag, neg_frag
    
    def get_subgroup_score_bin(self,smiles_list,solvent_list,inverse_transform=False, graph_id = None, key = 'se1'):
        results = self.get_scores(smiles_list, solvent_list, inverse_transform=inverse_transform, graph_id = graph_id, key = key)

        positive_frag_scores = {}
        negative_frag_scores = {}
        
        for c,smiles in enumerate(results):
            smiles,solvent = smiles
            score = results[(smiles,solvent)]
            if key in score and score[key].ndim != 2:
                score[key] = score[key].reshape(-1,1)
            frag = self.get_fragment(smiles)
            for i,f in enumerate(frag):
                f_smiles = f.smiles
                if key in score:
                    if f_smiles not in positive_frag_scores:
                        positive_frag_scores[f_smiles] = score[key][i]
                    else:
                        positive_frag_scores[f_smiles] = np.concatenate([positive_frag_scores[f_smiles], score[key][i]], axis=0)
            if (c+1)%1000==0:
                print(c+1)
        return positive_frag_scores, negative_frag_scores
    
    def get_scores(self, smiles_list, solvent_list, inverse_transform=False, graph_id = None, key = 'se1'):
        results = {}

        valid_smiles,valid_solvent, new_results = self._get_all_scores(smiles_list, solvent_list, inverse_transform=inverse_transform, graph_id = graph_id, key = key)
        for smiles, solvent, result in zip(valid_smiles,valid_solvent, new_results,):
            results[(smiles,solvent)] = result
        return results      
    
    def _get_all_scores(self, smiles_list, solvents_list, inverse_transform=False, graph_id = None, key = 'se1'):
        temp_loader,valid_smiles = self.prepare_temp_data(smiles_list, solvents_list, graph_id = graph_id)
        if len(valid_smiles) == 0:
            return [], [], []
        scores = self.tm.get_score(self.nm, temp_loader)
        scores = {k:scores[k].detach().cpu().numpy() for k in scores.keys()}
        results = []
        count=0
        for i, smiles in enumerate(valid_smiles['compound']):
            result={}
            result['vp']=scores['vp'][i]

            frag = self.get_fragment(smiles, get_index=True)
            result[key] = scores[key][count:count+len(frag)]
            count+=len(frag)
            results.append(result)
            
            
            result[key] = self.get_group_score(smiles, result[key],frag)
            self.save_data(smiles, result)
        if inverse_transform:
            for result in results:
                result= self.inverse_transform(result)

        return valid_smiles['compound'], valid_smiles['solvent'], results
    
    def inverse_transform(self, score, is_relative=False):
        """
        This function inverse transforms the score.

        Args:
            score (float): The score to inverse transform.
        
        Returns:
            float: The inverse transformed score.
        
        """
        if type(score) is dict:
            for k in score.keys():
                if len(score[k])==0:
                    continue
                if score[k].ndim == 1:
                    score[k] = score[k].reshape(-1,1)
                score[k] = self.scaler.inverse_transform(score[k]) - self.scaler.inverse_transform(np.array([[0]]))
        elif type(score) is np.ndarray:
            if score.ndim == 1:
                score = score.reshape(-1,1)
            if is_relative:
                score = self.scaler.inverse_transform(score) - self.scaler.inverse_transform(np.array([[0]]))
            else:
                score = self.scaler.inverse_transform(score)
        return score
    
    def get_fragment(self, smiles, get_index=False):
        frag = None#self.load_data(smiles, 'fragments')
        if frag is None:
            sculptor = self.dm.gg.sculptor
            frag = sculptor.fragmentation_with_condition(Chem.MolFromSmiles(smiles),draw=False,get_index = False)
            self.save_data(smiles, {'fragments': frag})
        if get_index:
            return [f.atoms for f in iter(frag)]
        return frag
    
    def prepare_temp_data(self, smiles_list, solvents = None, graph_id = None):
        if solvents is not None:
            valid_smiles = self.dm.init_temp_data(smiles_list, solvents, graph_id = graph_id)
        else:
             valid_smiles = self.dm.init_temp_data(smiles_list, graph_id = graph_id)
        temp_loader = self.dm.get_Dataloaders(temp=True)
        return temp_loader, valid_smiles

class SGCAnalyzer_v1p3(Analyzer.ISAAnalyzer_v1p3):
        
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
