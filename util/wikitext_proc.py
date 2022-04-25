import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

re_is_heading = re.compile('=.*?=')
stop_words = set(stopwords.words('english'))
stop_words.update('.', ',', '@', '-', ':', '(', ')', '?', '!', '\'', '+', '/')


def check_line(modified: str):
    modified = modified.strip()

    if not modified:
        return False
    if re_is_heading.search(modified):
        return False
    if len(modified) < 15:
        return False

    return True


def process_line(line: str):
    proc = line.replace('<unk>', '')
    proc = word_tokenize(proc)
    proc = [c.lower() for c in proc if c.lower() not in stop_words]

    return proc