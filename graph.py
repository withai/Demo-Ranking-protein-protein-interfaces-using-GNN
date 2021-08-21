import matplotlib.pyplot as plt

def show_results(pcomplex, scores_dockq):
    dockq_scores = scores_dockq[pcomplex]["dockq"]
    predicted_scores = scores_dockq[pcomplex]["pred_score"]

    num_decoys = range(0, len(dockq_scores), 1)

    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)

    ax1.scatter(num_decoys, dockq_scores, s=5, c='b', marker="s", label='DockQ score')
    ax1.scatter(num_decoys, predicted_scores, s=5, c='r', marker="o", label='Predicted score')

    plt.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=False)
    plt.xlabel("Decoys")
    plt.ylabel("DockQ score [0, 1]")
    plt.legend(loc='upper right')
    plt.show()