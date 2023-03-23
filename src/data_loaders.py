import numpy as np

def process_line(line: str) -> np.ndarray:

    line_without_comment = line.split('#')[0]
    line_without_comment = line_without_comment.strip()
    features = line_without_comment.split(sep=' ')
    score = float(features[0])
    qid = int(features[1].split(':')[1].strip())
    processed_features = list(map(lambda x: float(x.split(':')[1].strip()), features[2:]))


    return qid, score, np.array(processed_features)



def process_dataset(dataset: str):
    qids = []
    scores = []
    features = []

    lines = dataset.splitlines()

    for line in lines:
        qid, score, feature_vec = process_line(line)

        qids.append(qid)
        scores.append(score)
        features.append(feature_vec)

    return np.array(qids), np.array(scores), np.stack(features)


def get_dataset(path: str):
    with open(path, 'r') as file:
        return process_dataset(file.read())


