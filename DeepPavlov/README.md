# DeepPavlov modifications

The version of DeepPavlov that I used was 0.14. The corresponding modified files are contained in this folder: Python scripts were pesent in "\home\USER\\.local\lib\python3.7\site-packages\deeppavlov\models\kbqa", while "sparql_queries.json" was contained in "\home\USER\\.deeppavlov\downloads\wikidata" (a folder generated automatically during the first instantiation of DeepPavlov model). To facilitate the application of the same modifications in case the files will change in future versions of the model, in the following list each change and the reason regarding it are explained.

- "sparql_queries.json" has been modified because a couple of templates (15 and 18) didn't have the field "define_sorting_order: false", which caused errors if the predicted template of one of the input questions was one of them.
- "query_generator.py" has been modified adding the following code:
  - ```
    import os
    with open(os.environ['lc-quad_output_path'] + "queries_templates.txt", "a") as queries_file:
        queries_file.write(str(query) + "\n")
    ```
    in "query_parser" function, after the definition of "query" variable.
  - ```
    with open(os.environ['lc-quad_output_path'] + "candidate_outputs_lists.txt", "a") as queries_file:
        queries_file.write(str(candidate_outputs_list) + "\n")
    for i, candidate_answers in enumerate(candidate_outputs_list):
        for j, candidate_answer in enumerate(candidate_answers):
            for z, answer_component in enumerate(candidate_answer):
                if isinstance(answer_component, int):
                    candidate_outputs_list[i][j][z] = str(i) + "§" + str(j) + "§" + str(z) + "|int|" + str(answer_component)
                else:
                    candidate_outputs_list[i][j][z] = str(i) + "§" + str(j) + "§" + str(z) + "|" + answer_component
    ```
    in "query_parser" function, after the definition of "candidate_outputs_list" variable. Note that the appending operation for "candidate_outputs_lists.txt" file should be useless, since the indexes added in the following part can be used to get directly the candidate query; in any case, it has been reported for completeness.
  - ```
    with open(os.environ['lc-quad_output_path'] + "candidate_outputs.txt", "a") as queries_file:
        queries_file.write(str(candidate_outputs) + "\n")
    for i, candidate in enumerate(candidate_outputs):
        for j, candidate_component in enumerate(candidate):
            if isinstance(candidate_component, str) and "|" in candidate_component:
                parts = candidate_component.split('|')
                if len(parts) > 2 and parts[1] == "int":
                    parts[-1] = int(parts[-1])
                candidate_outputs[i][j] = parts[-1]
    ```
    in "query_parser" function, before the final return.
- "wiki_parser.py" has been modified adding the following code:
  - ```
    queries_complete_candidates = []
    ```
    in "\_\_call__" function, after the instantiation of the first two lists.
  - ```
    queries_complete_candidates.append([query[1], query[2]])
    ```
    in "\_\_call__" function, after the instruction "queries_candidates.append(query[1])".
  - ```
    with open(os.environ['lc-quad_output_path'] + "queries_candidates.txt", "a") as queries_file:
        queries_file.write(str(queries_complete_candidates) + "\n")
    ```
    in "\_\_call__" function, before the return contained in "query_execute" case.
  - ```
    if queries_complete_candidates:
        with open(os.environ['lc-quad_output_path'] + "queries_candidates.txt", "a") as queries_file:
            queries_file.write(str(queries_complete_candidates) + "\n")
    ```
    in "\_\_call__" function, before the final return; this part has been added in case "return_if_found" variable is false.
- "rel_ranking_bert_infer.py" has been modified replacing the instruction
  ```
  answers_with_scores = sorted(answers_with_scores, key=lambda x: x[-1], reverse=True)
  ```
  of "\_\_call__" function with the code
  ```
  answers_with_scores_and_indexes = []
  for i, tuple_answer in enumerate(answers_with_scores):
      answers_with_scores_and_indexes.append((i,) + tuple_answer)
  answers_with_scores_and_indexes = sorted(answers_with_scores_and_indexes, key=lambda x: x[-1], reverse=True)
  import os
  with open(os.environ['lc-quad_output_path'] + "answers_indexes.txt", "a") as answer_file:
      if len(answers_with_scores_and_indexes) > 0:
          answer_file.write(str(answers_with_scores_and_indexes[0][0]) + "\n")
      else:
          answer_file.write("-\n")
  answers_with_scores = []
  for tuple_answer in answers_with_scores_and_indexes:
      list_answer = list(tuple_answer)
      del list_answer[0]
      answers_with_scores.append(tuple(list_answer))
  ```