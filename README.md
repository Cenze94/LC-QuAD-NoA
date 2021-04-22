# LC-QuAD-NoA

This is the repository of "I know that I know nothing: Questions Answering with Knowledge Bases" thesis files. Biggest data, like models checkpoints, has to be downloaded from the links shown in the descriptions contained in the READMEs of each specific folder, which also explain their content.

- "[Models](Models)" contains the code for DeepPavlov execution, the four solutions training and testing, and their application to modify DeepPavlov results. It requires the files of datasets generated using "LC-QuAD-NoA" functions, and those obtained using the DeepPavlov version with the modified files contained in "DeepPavlov" folder. For code reproducibility, there are already LC-QuAD-NoA and DeepPavlov answers files.
- "[Cluster_universitario](Cluster_universitario)" contains the files specific for the University of Padua's Department of Mathemathics' Cluster.
- "[DeepPavlov](DeepPavlov)" contains the model modified files that I used during my thesis, and the explanation of which points have been modified, in case those files change over time.
- "[LC-QuAD-NoA](LC-QuAD-NoA)" contains the code for LC-QuAD-NoA dataset construction.
- "[NarrativeQA](NarrativeQA)" contains some files with the information I used for NarrativeQA dataset analysis.

For dataset generation and LSTM execution I used a local machine, while DeepPavlov required the computational power of the Department of Mathemathics' Cluster because of its high RAM demand. The other solutions have been executed with Colab notebooks. In the first two cases I used Python 3.7 with the requirements saved in "LC-QuAD-NoA" and "Cluster", respectively; in the last case I only added some libraries, through the code already present in the notebooks.