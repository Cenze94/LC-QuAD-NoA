from data_extractor import *
from rouge_l_calculator import rouge_l_calculation
from statistics import mean

narrative_qa_path = "../NarrativeQA/narrativeqa_completo/"
narrative_qa_qaps_sub_path = "qaps.csv"
narrative_qa_summaries_subpath = "third_party/wikipedia/summaries.csv"
hard_em_predictions_path = "../QA_Hard_EM/out/test_predictions.json"
multi_hop_predictions_path = "../CommonSenseMultiHopQA/NQACommonsense_preds.txt"
output_file_name = "predictions_answers_table.csv"

def main():
    # Get predictions, answers data and ROUGE-L scores
    hard_em_predictions = extract_Hard_EM_predictions(hard_em_predictions_path)
    multi_hop_predictions = extract_Multi_Hop_predictions(multi_hop_predictions_path)
    documents_ids, questions, first_answers, second_answers = extract_questions_answers_data(narrative_qa_path +
                                                                                           narrative_qa_qaps_sub_path)
    hard_em_first_answer_scores, hard_em_second_answer_scores, hard_em_best_answer_scores = rouge_l_calculation(
        first_answers, second_answers, hard_em_predictions)
    multi_hop_first_answer_scores, multi_hop_second_answer_scores, multi_hop_best_answer_scores = rouge_l_calculation(
        first_answers, second_answers, multi_hop_predictions)
    answer_best_model = []
    for i, hard_em_score in enumerate(hard_em_best_answer_scores):
        multi_hop_score = multi_hop_best_answer_scores[i]
        if hard_em_score >= multi_hop_score:
            answer_best_model.append("Hard EM")
        else:
            answer_best_model.append("Multi Hop")

    # Create csv file with results
    with open(output_file_name, mode='w', encoding='utf-8', newline='') as output_file:
        fieldnames = ["document_id", "question", "answer1", "answer2", "HEM_Prediction", "HEM_ROUGE-L_1", "HEM_ROUGE-L_2",
                      "HEM_ROUGE-L_best", "MH_Prediction", "MH_ROUGE-L_1", "MH_ROUGE-L_2", "MH_ROUGE-L_best",
                      "best_model"]
        output_writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter='|', quotechar='{')
        output_writer.writeheader()
        for i in range(len(questions)):
            output_writer.writerow({
                'document_id': documents_ids[i],
                'question': questions[i],
                'answer1': first_answers[i],
                'answer2': second_answers[i],
                'HEM_Prediction': hard_em_predictions[i],
                'HEM_ROUGE-L_1': hard_em_first_answer_scores[i],
                'HEM_ROUGE-L_2': hard_em_second_answer_scores[i],
                'HEM_ROUGE-L_best': hard_em_best_answer_scores[i],
                'MH_Prediction': multi_hop_predictions[i],
                'MH_ROUGE-L_1': multi_hop_first_answer_scores[i],
                'MH_ROUGE-L_2': multi_hop_second_answer_scores[i],
                'MH_ROUGE-L_best': multi_hop_best_answer_scores[i],
                'best_model': answer_best_model[i]
            })

    # Print averages
    print(mean(hard_em_best_answer_scores))
    print(mean(multi_hop_best_answer_scores))

if __name__ == "__main__":
    main()