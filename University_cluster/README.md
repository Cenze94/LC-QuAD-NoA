# Cluster

For the execution with the Cluster of Python scripts saved in [Models](Models) folder, I created a virtualenv of Python 3.7 with Anaconda, called "py37", installed the libraries of "[requirements](../Models/requirements.txt)", and used the "cluster_long" queue.

For each "job" file, the following information has to be modified:

- email;
- standard output and standard error directories;
- work directory.

"deeppavlov.job" has two possible executions, the second one runs DeepPavlov model with an entire dataset executing "execute_deeppavlov_model.py", while the first one runs the same model with a single question.