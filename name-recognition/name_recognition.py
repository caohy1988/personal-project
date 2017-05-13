import math
import pickle

model_data = pickle.load(open('gib_model.pki', 'rb'))

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '

pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])


def normalize(line):
    """ Return only the subset of chars from accepted_chars.
    This helps keep the  model relatively small by ignoring punctuation, 
    infrequenty symbols, etc. """
    return [c.lower() for c in line if c.lower() in accepted_chars]


def ngram(n, l):
    """ Return all n grams from l after normalizing """
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])


def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))


def valid_name(string):
    if not string:
        return 0
    if len(string) < 2:
        return 0
    if not string.isalpha():
        return 0
    string = string.lower()

    # using the pre-trained threshold to return 1 or 0
    score = avg_transition_prob(string, model_data['mat'])

    if score >= model_data['thresh']:
        return 1
    else:
        return 0


def main():
    valid_name('Haiyuan')
    valid_name('Jose')
    valid_name('koli')
    valid_name('fgg')
    valid_name('hg')
    valid_name('Ankit')
    valid_name('gfsdgdfsg')
    valid_name('gsdfgsf')


if __name__ == '__main__':
    main()
