from pathlib import Path
import pandas  as pd 
import matplotlib.pyplot as plt

def main():
    path_to_data = Path(__file__).parent / "data" 
    path_to_figures = Path(__file__).parent / "figures" 
    insomnia_set_variance = pd.read_csv(path_to_data/ "insmnia5_set_variance.csv").drop(["Unnamed: 0"],axis=1).mean(axis=1)
    #non_insomnia_set_variance = pd.read_csv(path_to_data/ "non-insmnia5_set_variance.csv").drop(["Unnamed: 0"],axis=1).mean(axis=1)
    #set_variance = pd.concat([insomnia_set_variance, non_insomnia_set_variance], axis=0)

    insomnia_soft_labels = pd.read_csv(path_to_data/ "insmnia5_hard_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1).mean(axis=1)
    #non_insomnia_soft_labels = pd.read_csv(path_to_data/ "non-insmnia5_hard_adam_mean_predictions.csv").drop(["Unnamed: 0"],axis=1).mean(axis=1)
    #insomnia_soft = pd.concat([insomnia_soft_labels, non_insomnia_soft_labels], axis=0)

    fig = plt.figure(figsize=(15,10))
    #plt.scatter(insomnia_set_variance, insomnia_soft_labels, marker=".", color="orange")
    plt.scatter(insomnia_set_variance, insomnia_soft_labels, marker=".", color="blue")
    plt.savefig(path_to_figures / "set_variance_soft_labels_relation.png")
    plt.close()

if __name__ == "__main__":
    main()