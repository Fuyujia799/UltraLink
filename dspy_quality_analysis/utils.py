import re
import string
import unicodedata
from collections import Counter
import dsp
from dsp.utils.utils import print_message

def F1_ultralink(prediction, answers_list):
    assert type(answers_list) == list

    return max(f1_score_ultralink(prediction, ans) for ans in answers_list)

def normalize_text_ultralink(s):

    parts = re.findall(r"<Classification>(.*?)<Issues>", s, flags=re.S)

    s = ' '.join(parts)  # Join the captured parts into a single string

    s = unicodedata.normalize('NFD', s)

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
            
    #print(white_space_fix(remove_articles(remove_punc(lower(s)))))
    #print("bbb")
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score_ultralink(prediction, ground_truth):
    prediction_tokens = normalize_text_ultralink(prediction).split()
    ground_truth_tokens = normalize_text_ultralink(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        # Unlike most tasks, QReCC and SQuAD-2.0 assign 1.0 in this edge case. We don't for uniformity.
        print_message(
            "\n#> F1 Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\n")

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    print(f1)
    return f1

def answer_match(prediction, answers, frac=1.0):
    # pred = example.prediction
    # answers = example.answers
    return F1_ultralink(prediction, answers)

def answer_exact_match(example, pred, trace=None, frac=1.0):
    assert(type(example.answer) is str or type(example.answer) is list)
    
    if type(example.answer) is str:
        return answer_match(pred.answer, [example.answer], frac=frac)
    else: # type(example.answer) is list
        return answer_match(pred.answer, example.answer, frac=frac)