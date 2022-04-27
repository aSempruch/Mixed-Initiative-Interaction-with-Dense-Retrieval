import faiss
import pickle as pkl
from tqdm import tqdm
import sys
import os

from util.wikitext_proc import check_line, process_line

# re_is_info_snippet = re.compile('\(.*?\).*?-')


if __name__ == '__main__':
    corpus = list()

    dataset_path = sys.argv[1]
    dataset_name = dataset_path.split('/')[-1]

    for file_name in os.listdir(dataset_path):
        file_path = f'{dataset_path}/{file_name}'

        with open(file_path, 'r') as f:
            file_len = sum([1 for _ in f])

        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=f'Processing {file_name}', total=file_len):
                if check_line(line):
                    processed = process_line(line)
                    if len(processed) > 1:
                        corpus.append(processed)

    output_path = f'data/{dataset_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'{output_path}/corpus.pkl', mode='wb') as f:
        pkl.dump(corpus, f)

    with open(f'{output_path}/collection.tsv', mode='w') as f:
        for idx, doc in enumerate(corpus):
            f.write(f'{idx}\t{" ".join(doc)}\n')

    doc_to_id_map = {" ".join(doc): idx for (idx, doc) in enumerate(corpus)}
    with open(f'{output_path}/doc_to_id_map.pkl', mode='wb') as f:
        pkl.dump(doc_to_id_map, f)
