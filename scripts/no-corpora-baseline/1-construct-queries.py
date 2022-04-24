import pandas as pd

train_data_split = pd.read_csv('ClariQ-master/data/train.tsv', sep="\t")
dev_data_split = pd.read_csv('ClariQ-master/data/dev.tsv', sep="\t")

train_data = pd.concat([train_data_split, dev_data_split], sort=False)

# initial_requests = pd.DataFrame(train_data['initial_request'].unique())
initial_requests = train_data[['topic_id', 'initial_request']].drop_duplicates().set_index('topic_id')

initial_requests.to_csv(
    'ClariQ-master/parsed/train-queries.tsv',
    sep="\t",
    header=None
)

# %% Construct Collection

question_bank = pd.read_csv('ClariQ-master/data/question_bank.tsv', sep="\t", index_col="question_id")
question_bank.dropna(inplace=True)

question_id_table = {q_idx: idx for idx, (q_idx, _) in enumerate(question_bank.iterrows())}

question_bank.reset_index(drop=True, inplace=True)
question_bank.to_csv(
    'ClariQ-master/parsed/question_bank.tsv',
    sep="\t",
    header=None
)


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
