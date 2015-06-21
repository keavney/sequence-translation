from log import log

import numpy
import random
from itertools import izip
from collections import defaultdict

from utils import *

def realtest(embed_src, embed_dst, model, X, Y):
    X_translations = [embed_src.match_sentence(list(reversed(x))) for x in X]
    Y_translations = [embed_dst.match_sentence(y) for y in Y]
    R = model.predict_batch(X, batch_size=len(X))
    #R_translations = [embed_dst.match_sentence(r) for r in R]
    R_translations = [[embed_dst.matchN(word, 10) for word in r] for r in R]
    i = 0
    for x, y, r, r_w in izip(X_translations, Y_translations, R_translations, R):
        print '=[ {0} ]================================='.format(i)
        print 'input: ', ' '.join(x)
        print 'label: ', ' '.join(y)
        print 'output:', ' '.join([w[0][0] for w in r])
        print 'input: ', x
        print 'label: ', y
        for o in r:
            print o
        print list(r_w)
        exit()
        i += 1

    return R_translations


def nptest(embed_src, embed_dst, model, X, Y):
    X_translations = [embed_src.match_sentence(list(reversed(x))) for x in X]
    Y_translations = [embed_dst.match_sentence(y) for y in Y]
    R = model.predict_batch(X, batch_size=len(X))
    R_translations = [embed_dst.match_sentence(r) for r in R]
    i = 0
    for x, y, r in izip(X_translations, Y_translations, R_translations):
        print i
        print ' '.join(x)
        print ' '.join(y)
        print ' '.join(r)
        i += 1

    return R_translations


def nptest_dict(embed_src, embed_dst, model, X, Y, M, DLs):
    # clip Y
    Y = [y[:len([mv[0] for mv in m if mv[0]])] for y, m in izip(Y, M)]

    X_translations = [embed_src.match_sentence(list(reversed(x))) for x in X]
    Y_translations = [embed_dst.match_sentence(y) for y in Y]

    R = model.predict_batch(X, batch_size=len(X))
    R_translations = [((mode, [embed_dst.match_sentence(r, w=D) for r in R]))
            for mode, D in DLs]
        
    all_acc = defaultdict(int)
    all_count = defaultdict(int)
    
    # TODO: clean this up
    RL = [mode for mode, r in R_translations]
    RZ = zip(*[r for mode, r in R_translations])
    R_iterable = [zip(RL, RZ[i]) for i in range(len(R_translations[0][1]))]

    # iterate
    i = 0
    for x, y, R in izip(X_translations, Y_translations, R_iterable):
        print i
        print ' '.join(x)
        print ' '.join(y)
        for mode, r in R:
            acc = sum([yy == rr for yy, rr in zip(y, r)]) / float(len(y))
            print mode, acc
            print ' '.join(r)
            all_acc[mode] = all_acc[mode] + acc
            all_count[mode] = all_count[mode] + 1
        print ""
        i += 1

    print all_acc
    print all_count
    all_sum = {k: v / float(all_count[k]) for k, v in all_acc.iteritems()}
    print all_sum

    return R_translations


def simpletest(embed_src, embed_dst, model, X, Y, n=5):
    X_translations = [embed_src.match_sentence(list(reversed(x))) for x in X]
    Y_translations = [embed_dst.match_sentence(y) for y in Y]
    R = model.predict_batch(X, batch_size=len(X))
    R_translations = [embed_dst.match_sentence(r) for r in R]
    i = 0
    for x, y, r in izip(X_translations, Y_translations, R_translations):
        print ' '.join(x), ',', ' '.join(y), '=>', ' '.join(z)
        i += 1

    return R_translations

# sets: expects tuples of (set name, input embeddings, label embeddings)
def datadump(embed_src, embed_dst, model, sets, epoch_no, size, oe_size, DLs):
    if oe_size is None:
        oe_size = embed_dst.embedding_size
    e_sets = []
    epoch = {'id': epoch_no, 'sets': e_sets}
    data = {'epochs': [epoch]}
    for name, X_emb, Y_emb in sets: 
        if size is not None:
            pairs = random.sample(zip(X_emb, Y_emb), size)
            X_emb = [x for x, y in pairs]
            Y_emb = [y for x, y in pairs]
        entries = []
        s = {'name': name, 'entries': entries}
        e_sets.append(s)
        R_emb = model.predict_batch(X_emb)
        for i, (x_emb, y_emb, r_emb) in enumerate(izip(X_emb, Y_emb, R_emb)):
            x = embed_src.match_sentence(list(reversed(x_emb)))
            y = embed_dst.match_sentence(y_emb)

            outputs = {}

            for mode, D, metric in DLs:
                r = embed_dst.match_sentence(r_emb, w=D, metric=metric)
                r_all = [embed_dst.matchN(o, oe_size, w=D, metric=metric) for o in r_emb]
                correct = sum((yy == rr for yy, rr in izip(y, r)))
                correct_pct = correct / float(len(y))
                rank_position = [pred_index(lambda mm: mm[0] == yy, rr) for (yy, rr) in izip(y, r_all)]
                rank_sorted = sorted(rank_position)
                outputs[mode] = {
                        'output': r,
                        'correct': correct, 
                        'correct_pct': correct_pct,
                        'rank_position': rank_position,
                        'rank_sorted': rank_sorted,
                        'dump': r_all,
                }

            entry = {
                    'id': i,
                    'input': x,
                    'label': y,
                    'outputs': outputs,
            }

            entries.append(entry)

        if len(entries) == 0:
            s['summary'] = {}
            continue

        summaries = {}
        for mode in entries[0]['outputs'].iterkeys():
            D = None
            for d in DLs:
                if d[0] == mode:
                    D = d
                    break
            avg_correct = mean([x['outputs'][mode]['correct'] for x in entries])
            avg_correct_pct = mean([x['outputs'][mode]['correct_pct'] for x in entries])
            #TODO: change these to account for clipping
            #avg_rank_position = [mean([x['outputs'][mode]['rank_position'][i] for x in entries]) for i in range(len(entries[0]['outputs'][mode]['rank_position']))]
            #avg_rank_sorted = [mean([x['outputs'][mode]['rank_sorted'][i] for x in entries]) for i in range(len(entries[0]['outputs'][mode]['rank_position']))]
            avg_rank_position = -1
            avg_rank_sorted = -1
            summary = {
                'D': D,
                'avg_correct': avg_correct,
                'avg_correct_pct': avg_correct_pct,
                'avg_rank_position': avg_rank_position,
                'avg_rank_sorted': avg_rank_sorted,
            }
            summaries[mode] = summary

        s['summary'] = summaries

    return epoch

def full_stats(round_stats, sets, DLs, model, sample_size=None, batch_size=8, log=lambda *_: None):
    if len(DLs) != 1:
        raise Exception("len(DLs) != 1 (only normal, L2 is supported)")
    DL = DLs[0]
    #fm = DL[2]
    emb = DL[2]

    stats = round_stats

    for name, sdict in sets.iteritems():
        X_emb = sdict['X_emb']
        Y_tokens = sdict['Y_tokens']
        summary = {}
        summaries = {DL[0]: summary}
        stats[name]['summary'] = summaries

        if sample_size is not None and sample_size > 0 and sample_size < len(X_emb):
            indices = random.sample(xrange(len(X_emb)), sample_size)
            X_emb = numpy.array([X_emb[i] for i in indices])
            Y_tokens = [Y_tokens[i] for i in indices]

        bs = 8

        correct_pct_total = 0
        correct_pct_size = 0

        while len(X_emb):
            X_emb_batch = X_emb[:bs]
            Y_tokens_batch = Y_tokens[:bs]
            R_emb_batch = model.predict_batch(X_emb_batch)

            R_tokens_batch = [emb.vectors_to_sentence(r) for r in R_emb_batch]

            correct = [sum((yy == rr for yy, rr in izip(y, r))) for y, r \
                    in izip(Y_tokens_batch, R_tokens_batch)]
            correct_pct = [c / float(len(y)) for c, y in izip(correct, Y_tokens_batch)]
            print '====================='
            #print R_emb_batch
            print R_emb_batch.shape
            print Y_tokens_batch
            print R_tokens_batch
            print correct
            print correct_pct

            correct_pct_total += sum(correct_pct)
            correct_pct_size += len(correct_pct)

            X_emb = X_emb[bs:]
            Y_tokens = Y_tokens[bs:]

        print 'correct_pct_total', name, correct_pct_total
        print 'correct_pct_size', name, correct_pct_size

        pct = correct_pct_total / float(correct_pct_size)
        print 'correct_pct', name, pct

        summary['pct_correct_pct'] = pct

    return stats


# sets: expects tuples of (set name, input embeddings, label embeddings, loss)
def setacc(embed_src, embed_dst, model, sets, epoch_no, td, size, DLs):
    e_sets = {}
    epoch = {'epoch': epoch_no, 'time': td, 'sets': e_sets}
    data = {'epochs': [epoch]}
    for name, X_emb, Y_emb, loss in sets: 
        if size is not None:
            ## too slow
            #pairs = random.sample(zip(X_emb, Y_emb), size)
            #X_emb = [x for x, y in pairs]
            #Y_emb = [y for x, y in pairs]
            indices = [random.randint(0, len(X_emb)-1) for i in xrange(size)]
            X_emb = [X_emb[i] for i in indices]
            Y_emb = [Y_emb[i] for i in indices]
        entries = []
        s = {'loss': loss}
        e_sets[name] = s
        R_emb = model.predict_batch(X_emb)
        for i, (x_emb, y_emb, r_emb) in enumerate(izip(X_emb, Y_emb, R_emb)):
            x = embed_src.match_sentence(list(reversed(x_emb)))
            y = embed_dst.match_sentence(y_emb)

            outputs = {}

            for mode, D, metric in DLs:
                r = embed_dst.match_sentence(r_emb, w=D, metric=metric)
                correct = sum((yy == rr for yy, rr in izip(y, r)))
                correct_pct = correct / float(len(y))
                outputs[mode] = {
                        'output': r,
                        'correct': correct, 
                        'correct_pct': correct_pct,
                }

            entry = {
                    'id': i,
                    'input': x,
                    'label': y,
                    'outputs': outputs,
            }

            entries.append(entry)

        if len(entries) == 0:
            s['summary'] = {}
            continue

        summaries = {}
        for mode in entries[0]['outputs'].iterkeys():
            D = None
            for d in DLs:
                if d[0] == mode:
                    D = d
                    break
            avg_correct = mean([x['outputs'][mode]['correct'] for x in entries])
            avg_correct_pct = mean([x['outputs'][mode]['correct_pct'] for x in entries])
            summary = {
                'D[1]': D[1],
                'avg_correct': avg_correct,
                'avg_correct_pct': avg_correct_pct,
            }
            summaries[mode] = summary

        s['summary'] = summaries

    return epoch

