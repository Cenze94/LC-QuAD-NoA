# NarrativeQA

This folder contains some data regarding my analisys of [https://github.com/deepmind/narrativeqa](NarrativeQA).

- "Answers_extractor" contains the Python scripts I used to build the final table with all the results and the relative ROUGE-L value for each of the two ground truth answers; "main.py" builds the final "predictions_answers_table.csv" file using the functions contained in "data_extractor.py" to get the answers contained in "CommonSenseMultiHopQA" and "QA_Hard_EM" folders, and "rouge_l_calculator.py" to calculate each ROUGE-L value.
- "CommonSenseMultiHopQA" contains the answers of [https://github.com/yicheng-w/CommonSenseMultiHopQA](MultiHop) model.
- "QA_Hard_EM" contains the answers of [https://github.com/shmsw25/qa-hard-em](Hard EM) model.
- The remaining two text files are comments in Italian about some questions that I've analysed.