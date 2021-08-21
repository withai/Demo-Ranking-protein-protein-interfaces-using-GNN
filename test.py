import torch
import pickle
import os
import sys
import numpy as np

def _get_dockq_scores_file_name(pcomplex_pick_name, dataset):
    dataset_cat_path = os.path.join("data", dataset)
    dockq_file = os.path.join(dataset_cat_path, pcomplex_pick_name, "dockq", pcomplex_pick_name+".pkl")
    return dockq_file

def get_scores_dockq(model, scores, dataset):
    
    model_scores_dict = scores

    pcomplexes = list(model_scores_dict.keys())
    pcomplexes.sort()
    
    result = {}
    for pcomplex_name in pcomplexes:
        dockq_prot_path = _get_dockq_scores_file_name(pcomplex_name, dataset)
        
        result[pcomplex_name] = {}
        with open(dockq_prot_path, "rb") as f:
            dockq_dict = pickle.load(f)

            pcomplex_decoys = list(model_scores_dict[pcomplex_name].items())

            
            pcomplex_decoys.sort(key=lambda tup:tup[1], reverse=True)
            
            pcomplex_decoys_dockq = [dockq_dict[decoy_name[:-4]+".pdb"][0] for decoy_name, score in pcomplex_decoys]
            pcomplex_decoys_pred_scores = [predicted_score for decoy_name, predicted_score in pcomplex_decoys]
            
            result[pcomplex_name]["dockq"] = pcomplex_decoys_dockq
            result[pcomplex_name]["pred_score"] = pcomplex_decoys_pred_scores

    return result
            
    

def test(model, device, test_loader, dataset):
    model.eval()
    test_loss = 0
    mini_batches = 0

    model_scores = {}
    model_results = {}

    with torch.no_grad():
        for batch_idx, local_batch in enumerate(test_loader):
            mini_batch_target = []
            mini_batch_output = []

            for i, item in enumerate(local_batch):

                if(item["vertices"].size()[0] == 0):
                    if(logger is not None):
                        logger.error(item["name"])
                    else:
                        print("Error: " + str(item["name"]))
                    continue

                # Move graph to GPU.
                prot_name, decoy_name = item["name"]
                vertices = item["vertices"].to(device)
                nh_indices = item["nh_indices"].to(device)
                int_indices = item["int_indices"].to(device)
                nh_edges = item["nh_edges"].to(device)
                int_edges = item["int_edges"].to(device)


                model_input = (vertices, nh_indices, int_indices, nh_edges, int_edges)

                output = model(model_input)

                try:
                    model_scores[prot_name]
                except:
                    model_scores[prot_name] = {}
                
                model_scores[prot_name][decoy_name] = output.item()
        
        scores_dockq = get_scores_dockq(model, model_scores, dataset)   
            
        return scores_dockq