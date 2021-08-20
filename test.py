import torch
import pickle
import os
import sys
import numpy as np

def _get_dockq_scores_file_name(pcomplex_pick_name):
    dataset_cat_path = "data/test"
    dockq_file = os.path.join(dataset_cat_path, pcomplex_pick_name, "dockq", pcomplex_pick_name+".pkl")
    return dockq_file
    
def model_near_native_ranks(model, scores, top_n=20, dockq_thresh=0.65):
    model_scores_dict = scores

    pcomplexes = list(model_scores_dict.keys())
    pcomplexes.sort()

    total_top_20_decoys = 0
    total_hq_decoys = 0

    total_complexes = []

    near_native_ranks = []

    for pcomplex_name in pcomplexes:
        pcomplex_ranks = []

        is_valid_pcomplex = False
        
        dockq_prot_path = _get_dockq_scores_file_name(pcomplex_name)
        with open(dockq_prot_path, "rb") as f:
            dockq_dict = pickle.load(f)

            pcomplex_decoys = list(model_scores_dict[pcomplex_name].items())

            
            pcomplex_decoys.sort(key=lambda tup:tup[1], reverse=True)
            
            pcomplex_decoys_dockq = [dockq_dict[decoy_name[:-4]+".pdb"][0] for decoy_name, score in pcomplex_decoys]

            for i, value in enumerate(pcomplex_decoys):
                decoy_name, decoy_score = value
                if(pcomplex_decoys_dockq[i] > 0):
                    pcomplex_ranks.append((i+1, pcomplex_decoys_dockq[i], decoy_name))
                    is_valid_pcomplex = True
        
        if(len(pcomplex_ranks) > 0):
            pcomplex_ranks.sort(key=lambda tup:tup[1], reverse=True)
            near_native_ranks.append(pcomplex_ranks[0][0])

        if(is_valid_pcomplex):
            total_complexes.append(pcomplex_name)

    model_ranks = [rank for rank in near_native_ranks if(rank != None)]

    top_n_present = 0
    for rank in model_ranks:
        if(rank<=top_n):
            top_n_present += 1

    return top_n_present, len(model_ranks)

def test(model, device, test_loader, threshold, top_ns):
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
                    
                

        top_n_results = []
        for top_n in top_ns:
            top_n_near_native, total_near_native_complexes = model_near_native_ranks(
                model, model_scores, top_n=top_n, dockq_thresh=threshold)
            
            top_n_results.append(total_near_native_complexes)
                    
            
        return top_n_results