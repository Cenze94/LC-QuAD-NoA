#from rouge import Rouge
from rouge_score import rouge_scorer
import re

# Calculate and get ROUGE_L scores of given predictions and answers
def rouge_l_calculation(first_answers, second_answers, predictions):
    """for i, prediction in enumerate(predictions):
        predictions[i] = prediction.casefold()
    for i, answer in enumerate(first_answers):
        first_answers[i] = answer.casefold()
    for i, answer in enumerate(second_answers):
        second_answers[i] = answer.casefold()"""
    """# Improve predictions making lowercase all letters, deleting punctuation and substituting apostrophe with a space
    for i, prediction in enumerate(predictions):
        prediction = prediction.casefold()
        prediction = prediction.replace(",", "")
        prediction = prediction.replace(":", "")
        prediction = prediction.replace('"', "")
        prediction = prediction.replace(";", "")
        prediction = prediction.replace(".", "")
        predictions[i] = prediction.replace("'", " ")

    # Improve answers making lowercase all letters, deleting punctuation and substituting apostrophe with a space
    for i, answer in enumerate(first_answers):
        answer = answer.casefold()
        answer = answer.replace(",", "")
        answer = answer.replace(":", "")
        answer = answer.replace('"', "")
        answer = answer.replace(";", "")
        answer = answer.replace(".", "")
        first_answers[i] = answer.replace("'", " ")

    for i, answer in enumerate(second_answers):
        answer = answer.casefold()
        answer = answer.replace(",", "")
        answer = answer.replace(":", "")
        answer = answer.replace('"', "")
        answer = answer.replace(";", "")
        answer = answer.replace(".", "")
        second_answers[i] = answer.replace("'", " ")"""

    """# Some answers might not contain letters or numbers, check them and in case replace the content with a word that
    # shouldn't be present in the original answers. It's not enough to check if the string is empty, because there could
    # be punctuation marks, which are not considered words by rouge calculation. Strings without letters or numbers
    # cause the raise of an exception, which should be avoided
    replacement_string = "a11b239EMPTY0llf"
    for i, row in enumerate(predictions):
        if not bool(re.search('[A-Za-z0-9]', row)):
            predictions[i] = replacement_string

    for i, row in enumerate(first_answers):
        if not bool(re.search('[A-Za-z0-9]', row)):
            first_answers[i] = replacement_string

    for i, row in enumerate(second_answers):
        if not bool(re.search('[A-Za-z0-9]', row)):
            second_answers[i] = replacement_string"""

    """# Calculate ROUGE scores
    rouge = Rouge()
    first_scores = rouge.get_scores(first_answers, predictions)
    second_scores = rouge.get_scores(second_answers, predictions)

    # Get ROUGE-L scores
    first_answer_scores = []
    second_answer_scores = []
    for i in range(len(first_scores)):
        first_answer_scores.append(first_scores[i]["rouge-l"]["f"] * 100)
        second_answer_scores.append(second_scores[i]["rouge-l"]["f"] * 100)"""

    scorer = rouge_scorer.RougeScorer(['rougeL'])
    first_answer_scores = []
    second_answer_scores = []
    for i in range(len(predictions)):
        first_answer_scores.append(scorer.score(first_answers[i], predictions[i]).get('rougeL').fmeasure * 100)
        second_answer_scores.append(scorer.score(second_answers[i], predictions[i]).get('rougeL').fmeasure * 100)

    # Prepare and return the list of best scores
    best_answer_scores = []
    for i, fs in enumerate(first_answer_scores):
        if fs >= second_answer_scores[i]:
            best_answer_scores.append(fs)
        else:
            best_answer_scores.append(second_answer_scores[i])

    return first_answer_scores, second_answer_scores, best_answer_scores