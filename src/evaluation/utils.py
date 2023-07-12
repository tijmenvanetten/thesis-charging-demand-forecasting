import json 
from collections import defaultdict

def load_scores_from_json(file, metric='MAE'):
    with open(file) as f:
        data = json.load(f)
    
    scores = defaultdict(list)
    for train_length, results in data['results'].items():
        scores['train_length'].append(train_length)
        for result in results:
            scores[result['model']].append(result['scores'][metric])
    return scores

    