# Cluster

Python scripts saved here are the same of [Models](Models) folder. For their execution with the Cluster, I created a virtualenv of Python 3.7 with Anaconda, called "py37", installed the libraries of "[requirements](../Models/requirements.txt)", and used the "cluster_long" queue. Therefore, for each "job" file, the following information has to be modified:

- email;
- standard output and standard error directories;
- work directory.

"deeppavlov.job" has two possible executions, the second one runs DeepPavlov model for each "LC-QuAD_2_train_balanced.json" question (executing "execute_deeppavlov_model.py" file), while the first one runs the same model with a single question.