# Models

To be able to run the scripts contained in this folder, you must have performed the instructions contained in [DeepPavlov](../DeepPavlov) folder.

## Code reproducibility

To get DeepPavlov answers with a specific dataset file, change the path contained in

```
with open("data/LC-QuAD_2_train_balanced.json", "r") as json_file:
```

line of "load_questions" function of "execute_deeppavlov_model.py". Answers and additional text files will be saved in "output" folder. Documents used for thesis calculations of statistics and classifiers execution are already saved in that folder, so you don't need to get them again to perform the next steps.

Metrics calculations are performed with the "save_model_accuracy_statistics" function of "check_deeppavlov_model_functions.py" script. It requires three inputs: the set file directory, the answers document path and the name of the output file, which usually is contained in "data" folder and contains the "statistics" substring. There are three optional inputs, i.e. the list of additional DeepPavlov files (described [here](#details-of-deeppavlov-execution)) directories, the classifier predictions file path and the classifier set file directory (that is used to find what questions have been predicted, since usually the set file is not the version obtained after the application of embeddings filter). If the optional files are not provided, then by convention each DeepPavlov paper answer, i.e. the answer obtained using the paper method of accuracy calculation, is set to False (not correct), and each prediction is considered False. The fields of statistics files 

## Details of DeepPavlov execution

DeepPavlov model execution creates 6 files that are needed for the next steps; these documents, identified by the substring in quotes, are the following.

- "deeppavlov_answers": contains DeepPavlov model answers;
- "queries_templates": contains the predicted templates;
- "queries_candidates": contains all candidate queries;
- "candidate_outputs_lists": contains the answers of every candidate query; this should be a useless file, since "candidate_outputs" indexes can be used to directly extract the corresponding candidate query, but I reported it anyway because it's created;
- "candidate_outputs": contains the non-empty answers of candidate queries, with indexes to locate the corresponding one;
- "answers_indexes": contains the index of final answers, to get them from "candidate_outputs".

Since in some of these files there may be multiple lines associated with the same question, "execute_deeppavlov_model.py" adds to all files, for each input question, a new line containing the expression "-|-", which is handled by the scripts of the following steps.

## Statistics file

There are two additional support functions in "check_deeppavlov_model_functions.py": the first one, "filter_test_questions", filters the test set from questions without English answers (if not already done [before](../LC-QuAD-NoA)), while the second one, "add_answers_to_file", adds up to 50 possible answers to the remaining questions of the input set.

The fields of a statistic file are the following:

- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 
- "": 

## Embeddings DB preparation

The script for embeddings DB construction is "db_embeddings_constructor.py", containing three main instructions. The first one is "embeddings_filter", which excludes the embeddings of languages different from English; since it creates a very large file, an I/O error could be caused: if this happens, the execution continues on a new file generated automatically. The second one is "tsv_to_sql", that takes in input the list of filtered embeddings files and creates the final DB; it has a second input, which is the number of embeddings processed at the same time, and determines the execution time of the function. In the end, the last one is "test_sql", that simply tries to run a query with the created DB. The first two instructions execution requires a large amount of time, even a few days.

The embeddings used for this work can be downloaded [here](wikidata_translation_v1.tsv.gz).