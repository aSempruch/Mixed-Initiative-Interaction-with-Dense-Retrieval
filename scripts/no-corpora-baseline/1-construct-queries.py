import pandas as pd
import pickle as pkl

for split in ('train', 'dev', 'test'):

    train_data = pd.read_csv(f'ClariQ-master/data/{split}.tsv', sep="\t")

    initial_requests = train_data[['topic_id', 'initial_request']].drop_duplicates().set_index('topic_id')

    initial_requests.to_csv(
        f'ClariQ-master/parsed/{split}-queries.tsv',
        sep="\t",
        header=None
    )

# %% Construct Collection and Question ID Map

question_bank = pd.read_csv('ClariQ-master/data/question_bank.tsv', sep="\t", index_col="question_id")
question_bank.dropna(inplace=True)

question_id_table = {q_idx: idx for idx, (q_idx, _) in enumerate(question_bank.iterrows())}

question_bank.reset_index(drop=True, inplace=True)
question_bank.to_csv(
    'ClariQ-master/parsed/question_bank.tsv',
    sep="\t",
    header=None
)

question_id_table_rev = {val: key for (key, val) in question_id_table.items()}
with open('ClariQ-master/parsed/question_id_map.pkl', mode='wb') as f:
    pkl.dump(question_id_table_rev, f)

# %% Construct triples

triples = list()

with open('ClariQ-master/parsed/triples-no-corpora-random-negatives.jsonl', 'w') as f:
    # f.write('["id", "positive", "negative"]\n')
    for request_idx, initial_request in initial_requests.iterrows():
        positives = train_data[
            train_data.initial_request == initial_request[0]
        ]['question_id'].map(question_id_table).dropna().astype(int)
        negatives = train_data[
            (train_data['initial_request'] != initial_request[0])
            & (~train_data['question_id'].isin(positives))
        ]['question_id'].map(question_id_table).dropna().astype(int)

        for positive in positives:
            random_negative = negatives.sample(1).values
            f.write(f'[{request_idx}, {positive}, {random_negative[0]}]\n')
