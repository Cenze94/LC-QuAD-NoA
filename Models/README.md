# Models

To be able to run the scripts contained in this folder, you must have performed the instructions contained in [DeepPavlov](../DeepPavlov) folder.

## Code reproducibility

To get DeepPavlov answers with a specific dataset file, change the path contained in

```
with open("data/LC-QuAD_2_train_balanced.json", "r") as json_file:
```

line of "load_questions" function of "execute_deeppavlov_model.py". Answers and additional text files will be saved in "output" folder. Documents used for thesis calculations of statistics and classifiers execution are already saved in that folder, so you don't need to get them again to perform the next steps.

Metrics calculations are performed with the "save_model_accuracy_statistics" function of "check_deeppavlov_model_functions.py" script. It requires three inputs: the set file directory, the answers document path and the name of the output file, which usually is contained in "data" folder and contains the "statistics" substring. There are three optional inputs, i.e. the list of additional DeepPavlov files (described [here](#details-of-deeppavlov-execution)) directories, the classifier predictions file path and the classifier set file directory (that is used to find what questions have been fed to the classifier, since usually the set file, whose directory is used as the first input of this function, is not the version obtained after the application of embeddings filter). If the optional files are not provided, then by convention each DeepPavlov paper answer, i.e. the answer obtained using the paper method of accuracy calculation, is set to False (not correct), and each prediction is considered False. DeepPavlov files used for this work are already present, except for candidate queries one; it can be downloaded from [here](Models.rar), together with classifiers checkpoints and the test set, already filtered from questions without groundtruth embeddings, with candidate queries DeepPavlov embeddings.

Before executing the dataset filtering of questions without embeddings of at least a groundtruth entity or property, we need to download the [embeddings DB](dataset_embeddings.rar). After that, "preprocess_questions_file" function of "embeddings_filter_dataset.py" script can be executed to apply the filtering and to save additional data for each question, like DeepPavlov candidate queries embeddings. Since the latter requires a lot of memory, train set can be constructed without them, adding an additional substring and a "False" value. Template encodings can be obtained by executing the functions "prepare_encodings" and "save_encodings", passing a "False" value to distinguish the train set from the test set. Finally, we need to get a validation set executing the function "get_validation_set", giving in input also the percentage of validation set questions and a seed.

Classifiers files are contained in "Models" folder. "BERT questions", "BERT questions answers" and "Solution questions embeddings" models have been trained and executed with Colab, using "bert_question/bert_question_tf.ipynb", "bert_question_answer/bert_question_answer_tf.ipynb" and "complex_model/bert_question_embeddings_tf.ipynb" notebooks, respectively. Instead "LSTM embeddings" have been trained and executed with a personal machine with "lstm_embeddings_gold_model/lstm_embeddings_gold_model.py" file. Predictions are saved in a file called "model_predictions.txt", which is contained in the "output" folder of each classifier specific folder. In the end, "check_deeppavlov_model_functions.py" have been executed for every classifier, as described above, in order to get the final results.

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

There are two additional support functions in "check_deeppavlov_model_functions.py": the first one, "filter_test_questions", filters the test set from questions without English answers (if not already done [before](../LC-QuAD-NoA#execution-reproduction)), while the second one, "add_answers_to_file", adds up to 50 possible answers to the remaining questions of the input set.

A statistics file contains the information per generation template and in total. Besides there is an additional subdivision, called "without english answers", that contains all the questions without answers with an English label; it is useful if for some reason the dataset filtering fails, and so to handle cases that can cause errors. The fields of the other statistics file subdivisions are the following:

- "right answers answerable": number of answerable questions correctly answered by DeepPavlov model;
- "not found answerable": number of answerable questions answered by DeepPavlov model with "Not Found";
- "wrong answers answerable": number of answerable questions wrongly answered by DeepPavlov model;
- "right answers paper answerable": number of answerable questions correctly answered by DeepPavlov model using its paper calculation method;
- "wrong answers paper answerable": number of answerable questions wrongly answered by DeepPavlov model using its paper calculation method;
- "right prediction wrong answer number answerable": number of answerable questions wrongly answered by DeepPavlov model and predicted answerable by the classifier;
- "wrong prediction right answer number answerable": number of answerable questions correctly answered by DeepPavlov model and predicted not answerable by the classifier;
- "total number answerable": number of answerable questions;
- "operation types statistics": list of statistics regarding unanswerable questions divided by the type of operation used to generate them;
- "not found not answerable": number of unanswerable questions correctly answered by DeepPavlov model;
- "wrong answers not answerable": number of unanswerable questions wrongly answered by DeepPavlov model;
- "right answers paper not answerable": number of unanswerable questions correctly answered by DeepPavlov model using its paper calculation method;
- "wrong answers paper not answerable": number of unanswerable questions wrongly answered by DeepPavlov model using its paper calculation method;
- "right prediction wrong answer number not answerable": number of unanswerable questions wrongly answered by DeepPavlov model and predicted unanswerable by the classifier;
- "wrong prediction right answer number not answerable": number of unanswerable questions correctly answered by DeepPavlov model and predicted answerable by the classifier;
- "total number not answerable": number of unanswerable questions;
- "total number": number of questions;
- "accuracy answerable": the number of answerable questions correctly answered by DeepPavlov model over the total number of answerable questions, i.e. it is the recall;
- "accuracy paper answerable": the number of answerable questions correctly answered by DeepPavlov model using its paper calculation method over the total number of answerable questions, i.e. it is the recall paper;
- "accuracy with predictions answerable": the number of answerable questions correctly answered by DeepPavlov model over the total number of answerable questions, calculated after applying the predictions to modify model answers and considering the "I don't know" cases as correct answers;
- "accuracy not answerable": the number of unanswerable questions correctly answered by DeepPavlov model over the total number of unanswerable questions, i.e. it is the specificity;
- "accuracy paper not answerable": the number of unanswerable questions correctly answered by DeepPavlov model using its paper calculation method over the total number of unanswerable questions, i.e. it is the specificity paper;
- "accuracy with predictions not answerable": the number of unanswerable questions correctly answered by DeepPavlov model over the total number of unanswerable questions, calculated after applying the predictions to modify model answers;
- "final accuracy": the number of questions correctly answered over the total number of questions;
- "final accuracy paper": the number of questions correctly answered over the total number of questions, using DeepPavlov paper calculation method;
- "final accuracy with answerable predictions": the number of questions correctly answered over the total number of questions, calculated after applying the "answerable question" predictions to modify model answers and considering the "I don't know" cases as correct answers;
- "final accuracy with not answerable predictions": the number of questions correctly answered over the total number of questions, calculated after applying the "unanswerable question" predictions to modify model answers;
- "final accuracy with all predictions": the number of questions correctly answered over the total number of questions, calculated after applying all predictions to modify model answers and considering the "I don't know" cases as correct answers;
- "final accuracy answerable predictions no I don't know": the number of questions correctly answered over the total number of questions, calculated after applying the "answerable question" predictions to modify model answers and considering the "I don't know" cases as neutral answers;
- "final accuracy all predictions no I don't know": the number of questions correctly answered over the total number of questions, calculated after applying all predictions to modify model answers and considering the "I don't know" cases as neutral answers.

## Embeddings DB preparation

The script for embeddings DB construction is "db_embeddings_constructor.py", containing three main instructions. The first one is "embeddings_filter", which excludes the embeddings of languages different from English; since it creates a very large file, an I/O error could be caused: if this happens, the execution continues on a new file generated automatically. The second one is "tsv_to_sql", that takes in input the list of filtered embeddings files and creates the final DB; it has a second input, which is the number of embeddings processed at the same time, and determines the execution time of the function. In the end, the last one is "test_sql", that simply tries to run a query with the created DB. The first two instructions execution requires a large amount of time, even a few days.

The embeddings used for this work can be downloaded from [here](wikidata_translation_v1.tsv.gz).

## Dataset filtering from questions without embeddings

There is an additional utility function in "embeddings_filter_dataset.py" script, called "count_answerable_and_unanswerable_questions", which is used to get the number of dataset answerable and unanswerable questions.

## Classifiers

Inside the "models" folder there are 6 different folders:

- "bert_question" contains the files of the model based on BERT and fed with questions;
- "bert_question_answer" contains the files of the model based on BERT and fed with questions and answers;
- "complex_model" contains the files of the model based on BERT and LSTM, which inputs are questions, embeddings and templates encoding;
- "lstm_embeddings_gold_gold" contains the files of an unofficial model based on LSTM, that is trained and tested with groundtruth embeddings;
- "lstm_embeddings_gold_model" contains the files of the model based on LSTM, trained with groundtruth embeddings and tested with DeepPavlov candidate queries embeddings;
- "lstm_embeddings_model_model" contains the files of a sketchy model based on LSTM, that is trained and tested with DeepPavlov candidate queries embeddings.

Each folder contains an "output" folder, used to save predictions in "model_predictions.txt" file and the model last checkpoint; the checkpoints saved in the repository are those used for this work. "lstm_embeddings_gold_gold" contains a working model, that wasn't included in the thesis because it isn't very interesting. "lstm_embeddings_model_model" has been built to be trained with the candidate query of questions with a DeepPavlov answer, so with a smaller dataset; it is still present in the repository in case it helps to apply some similar idea. "bert_question" and "bert_question_answer" contain also a Python script, with an outdated version of the model written with PyTorch. Besides, "lstm_embeddings_gold_gold" contains also a Python script, "lstm_embeddings_gold_gold_keras.py", with an outdated version of the model written in Keras, and another script, "lstm_embeddings_gold_gold_multiple.py", with a system that trains a model for each generation template, and so without using the corresponding one-hot encodings. Finally, all LSTM models contain a function, called "preprocess_questions", that was used to build a specific dataset based on the type of embeddings needed (for instance, for "lstm_embeddings_model_model" all questions without a DeepPavlov answer were excluded), and for this reason it is obsolete.