# LC-QuAD-NoA

"entities_and_properties" folder contains the lists used for difficulty 1 operations. Instead, "dataset" folder contains all dataset files: "train.json" and "test.json" are the original LC-QuAD 2.0 files, while the other documents have been generated from them.

## Execution reproduction

To recreate LC-QuAD-NoA dataset, first delete "train_generated.json" and "test_generated.json" files, contained in "dataset" folder. After that, to filter LC-QuAD 2.0 from unanswerable answers, execute the instruction

```
python all_ops_generator.py --type "db_filtering"
```

to generate the files "train_filtered.json" and "test_filtered.json"; this command may take a few hours to execute. Questions have been generated executing a specific instruction for each template type, and saved in files "train_generated.json" and "test_generated.json". The generic version of this instruction is

```
python all_ops_generator.py --type "template_type" --n "total_questions_number"
```

where "template_type" is the generation template name, like "right_subgraph_2", and "total_questions_number" is the total number of questions to generate, divided proportionally between the train and the test set; its default value is 2000, which has been the number used officially. Keep in mind that due to approximation errors, the generated questions will actually be a little less. Besides, generated questions "question" and "paraphrased_question" fields have to be checked and corrected manually, as they often fail to be changed automatically. Finally, with the instruction

```
python all_ops_generator.py --type "create_balanced_dataset"
```

the original questions corresponding to those generated are extracted and the new files "train_balanced.json" and "test_balanced.json" are created.

## Files description

- Files with the "_questions_generator" substring are generation template classes specific scripts that contain the functions for each operation type and difficulty. In "simple_question_left_questions_generator.py" and "right_subgraph_questions_generator.py" there is an additional function, containing the "_template_generation" substring, which creates a new answerable question of the same template; it is part of an idea that was later abandoned, that is, of obtaining unanswerable questions from generated elements. There are also two additional operations in "simple_question_left_questions_generator.py", i.e. "simple_question_left_generic_generation" and "simple_question_left_generic_2_generation", which creates a new question and modifies it using randomly operations with entities or relations of difficulty 1 and 3, respectively.
- "questions_generator.py" contains the common code for all operation types, and the common utility functions.
- "main.py" has been the first file used to design LC-QuAD-NoA automatic procedure; it creates new questions using the examples contained in "entities_and_properties/Examples_templates.json", and saving the generated elements in file "entities_and_properties/Generated_questions.json". The latter contains also the updated "uid", which is also maintained and used by "all_ops_generator.py" script. "main.py" file can be executed with the command

  ```
  python main.py --type "template_type" --op "operation_type" --n "generated_questions_number"
  ```

  where "template_type" is the generation template name, like "right_subgraph_2", "operation_type" is the operation name, i.e. "entity", "relation", "entity_2", "relation_2", "entity_3" or "relation_3", and "generated_questions_number" is the number of questions to generate from the same original element.
- "wikidata_ids_extractor.py" contains the functions to extract the lists used in the operations of difficulty 1, from which a random element is selected.
- "all_ops_generator.py" is the script used for the final construction of LC-QuAD-NoA dataset. Some possible instructions have already been explained in [Execution reproduction](https://github.com/Cenze94/LC-QuAD-NoA/tree/master/LC-QuAD-NoA#execution-reproduction) section; there are two additional utility commands for the "type" argument, i.e. "check_original_questions" and "check_calculations". The first one checks if there are questions that have been generated from unanswerable elements: this may happen for example because during the generation of the new dataset Wikidata could have been modified. Since these are usually a few cases, the UIDs of original and generated questions are reported, in order to manually create new questions to replace them. In my case I simply replaced the corresponding example in "entities_and_properties/Examples_templates.json" with a question taken from the end of the filtered train or test set, and generated the element for replacement with "main.py" script. "check_calculations" is a simple check of the calculations of the number of questions to generate, which can be used to verify that the calculations function works as expected, and therefore the different number of questions actually generated is due to approximation errors.