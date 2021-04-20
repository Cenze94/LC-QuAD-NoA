# Models

To be able to run the scripts contained in this folder, you must have performed the instructions contained in [DeepPavlov](../DeepPavlov) folder.

## Code reproducibility

To get DeepPavlov answers with a specific dataset file, change the path contained in

```
with open("data/LC-QuAD_2_train_balanced.json", "r") as json_file:
```

line of "load_questions" function of "execute_deeppavlov_model.py". Answers and additional text files will be saved in "output" folder. Documents used for thesis calculations of statistics and classifiers execution are already saved in that folder, so you don't need to get them again to perform the next steps.



## Details of DeepPavlov execution

DeepPavlov model execution creates 6 files that are needed for the next steps; these documents are the following.

- "deeppavlov_answers": 