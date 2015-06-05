import sys
import argparse
import subprocess
import collections

import helpers
import output_dumps

from datetime import datetime
from log import log, log_to_file, stat

def main():
    # mapping of commands to handlers
    valid_commands = [
        ('create',      h_create),
        ('compile',     h_compile),
        ('train',       h_train),
        ('test',        h_test),
        ('export',      h_export),
        ('interactive', h_interactive),
    ]

    # create parser
    parser = argparse.ArgumentParser(description="LSTM Encoder Decoder",
            fromfile_prefix_chars='@')
    
    # global
    helpstr = "List of commands: " + ', '.join([x[0] for x in valid_commands])
    parser.add_argument('commands', type=str, nargs='+',
            help=helpstr)

    # data
    parser.add_argument('--train-src', dest='train_src', type=str,
            help="Training sentences for source (encoder) network")
    parser.add_argument('--train-dst', dest='train_dst', type=str,
            help="Training sentences for destination (decoder) network")
    parser.add_argument('--train-both', dest='train_both', type=str,
            help="Training sentences for both encoder and decoder network")
    parser.add_argument('--validation-src', dest='validation_src', type=str,
            help="Validation sentences for source (encoder) network")
    parser.add_argument('--validation-dst', dest='validation_dst', type=str,
            help="Validation sentences for destination (decoder) network")
    parser.add_argument('--validation-both', dest='validation_both', type=str,
            help="Test sentences for both encoder and decoder network")
    parser.add_argument('--test-src', dest='test_src', type=str,
            help="Test sentences for source (encoder) network")
    parser.add_argument('--test-dst', dest='test_dst', type=str,
            help="Test sentences for destination (decoder) network")
    parser.add_argument('--test-both', dest='test_both', type=str,
            help="Test sentences for both encoder and decoder network")

    # compile parameters
    parser.add_argument('--embedding-size', dest='embedding_size', type=int,
            help="Embedding vector size")
    parser.add_argument('--layers', dest='layer_count', type=int,
            help="Network layer count")
    parser.add_argument('--max-sentence-length', dest='maxlen', type=int,
            help="Maximum sentence length")
    parser.add_argument('--optimizer', dest='optimizer', type=str,
            default='adagrad',
            help="Optimizer type (rmsprop, sgd, adadelta, adagrad)")
    parser.add_argument('--compile-train', dest='compile_train', type=str,
            default='True',
            help="Compile training functions for model")

    # train parameters
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            default=16,
            help="Training batch size")
    parser.add_argument('--validation-skip', dest='validation_skip', type=float,
            default=10,
            help="Amount of epochs to skip before outputting validation translations")
    parser.add_argument('--snapshot-skip', dest='snapshot_skip', type=float,
            default=10,
            help="Amount of epochs to skip between snapshots")
    parser.add_argument('--lr-encoder', dest='lr_encoder', type=float,
            default=None,
            help="Learning rate for encoder")
    parser.add_argument('--lr-decoder', dest='lr_decoder', type=float,
            default=None,
            help="Learning rate for decoder")
    parser.add_argument('--lr-both', dest='lr_both', type=float,
            default=None,
            help="Learning rate for both")

    # trianing thresholds
    parser.add_argument('--epochs', dest='epochs', type=int,
            default=None,
            help="Cutoff for training (number of epochs)")
    parser.add_argument('--error', dest='error', type=float,
            default=None,
            help="Cutoff for training (test and validation error)")
    parser.add_argument('--seconds', dest='seconds', type=float,
            default=None,
            help="Cutoff for training (total seconds elapsed)")
    parser.add_argument('--loss', dest='loss', type=float,
            default=None,
            help="Cutoff for training (test and validation loss)")

    # test parameters
    parser.add_argument('--format', dest='test_format', type=str,
            default='',
            help="Test output format (options: pairs (default), simple, complex)")

    # logging
    parser.add_argument('--log-global', dest='log_glob', type=str,
            help="Log file for all output")
    parser.add_argument('--log-info', dest='log_info', type=str,
            help="Log file for updates (no data dumps)")
    parser.add_argument('--log-stat', dest='log_stat', type=str,
            help="Log file for stats (validation accuracy, etc)")

    # models
    parser.add_argument('--embedding-src', dest='embedding_src', type=str,
            help="Input filename for src embedding")
    parser.add_argument('--embedding-dst', dest='embedding_dst', type=str,
            help="Input filename for dst embedding")
    parser.add_argument('--embedding-both', dest='embedding_both', type=str,
            help="Input filename for both embedding")
    parser.add_argument('--output-embedding-src', dest='output_embedding_src', type=str,
            help="Output filename for src embedding")
    parser.add_argument('--output-embedding-dst', dest='output_embedding_dst', type=str,
            help="Output filename for dst embedding")
    parser.add_argument('--output-embedding-both', dest='output_embedding_both', type=str,
            help="Output filename for both embedding")
    parser.add_argument('--compiled-model', dest='compiled_model', type=str,
            help="Input filename for compiled model")
    parser.add_argument('--output-compiled-model', dest='output_compiled_model', type=str,
            help="Output filename for compiled model")
    parser.add_argument('--fitted-model', dest='fitted_model', type=str,
            help="Input filename for fitted model")
    parser.add_argument('--output-fitted-model', dest='output_fitted_model', type=str,
            help="Output filename for fitted model")
    parser.add_argument('--model-weights', dest='model_weights', type=str,
            help="Input filename for model weights")
    parser.add_argument('--output-model-weights', dest='output_model_weights', type=str,
            help="Output filename for model weights")
    parser.add_argument('--output-snapshot-prefix', dest='output_snapshot_prefix', type=str,
            help="Output prefix for snapshots")

    
    args = parser.parse_args()

    # handle 'both' arguments here
    if args.train_both is not None:
        args.train_src = args.train_both
        args.train_dst = args.train_both
    if args.validation_both is not None:
        args.validation_src = args.validation_both
        args.validation_dst = args.validation_both
    if args.test_both is not None:
        args.test_src = args.test_both
        args.test_dst = args.test_both
    if args.embedding_both is not None:
        args.embedding_src = args.embedding_both
        args.embedding_dst = args.embedding_both
    if args.output_embedding_both is not None:
        args.output_embedding_src = args.output_embedding_both
        args.output_embedding_dst = args.output_embedding_both
    if args.lr_both is not None:
        args.lr_encoder = args.lr_both
        args.lr_decoder = args.lr_both

    # handle logs
    if args.log_glob is not None:
        log_to_file('glob', args.log_glob)
    if args.log_info is not None:
        log_to_file('info', args.log_info)
    if args.log_stat is not None:
        log_to_file('stat', args.log_stat)

    log("Loaded arguments")
    print args
    
    commands = map(str.lower, args.commands)

    cache = collections.defaultdict(lambda: None, {'commands': commands})
    
    # check that all commands are valid before executing
    for command in commands:
        if command not in map(lambda x: x[0], valid_commands):
            log("Parsed invalid command {0}: exiting".format(command))
            exit()
    
    # execute commands in order
    for command in commands:
        actions = filter(lambda x: x[0] == command, valid_commands)
        for name, handler in actions:
            log("Entering command: {0}".format(name))
            handler(cache, args)
            log("Finished command: {0}".format(name))
    
    log("Finished all commands, exiting")


# wrapper for args lookup that exits if arg wasn't provided
def get_required_arg(args, name):
    res = getattr(args, name)
    if res is None:
        caller = sys._getframe().f_back.f_code.co_name
        print "Error in {0}: parameter {1} not specified. (Did you forget a flag?) Exiting.".format(caller, name)
        exit(1)
    return res


# wrapper for embedding
def get_embedding(cache, args, name):
    res = cache[name]
    if res is None:
        embedding_filename = get_required_arg(args, name)
        res = helpers.create_embed(embedding_filename)
        cache[name] = res
    return res

# wrapper for fitted model
def get_fitted_model(cache, args):
    # load model
    input_model = args.fitted_model
    input_weights = args.model_weights
    compiled_model = args.compiled_model
    model = None
    if input_model is not None:
        model = helpers.import_model(input_model)
    elif input_weights is not None:
        model = cache['compiled_model']
        if compiled_model is not None:
            model = helpers.import_model(compiled_model)
        if model is None:
            print "Error loading fitted model: weights were provided, but a model was not compiled. Exiting"
        weights = helpers.import_weights(input_weights)
        model.set_weights(weights)
    elif 'fitted_model' in cache:
        model = cache['fitted_model']
    else:
        print "Error loading fitted model: no fitted model provided: exiting"
        exit()
    return model


# command handlers
# (TODO: create list of arg dependencies)
def h_create(cache, args):
    e = 'word2vec'

    # check for exec existance
    try:
        subprocess.check_call('{0} > /dev/null 2>&1'.format(e), shell=True)
    except subprocess.CalledProcessError:
        print "Error in create: couldn't call process {0}: exiting".format(e)
        exit()
    except OSError:
        print "Executable {0} not found in path: exiting".format(e)
        exit()

    # get args
    infile_src = get_required_arg(args, 'train_src')
    infile_dst = get_required_arg(args, 'train_dst')
    outfile_src = get_required_arg(args, 'output_embedding_src')
    outfile_dst = get_required_arg(args, 'output_embedding_dst')
    embedding_size = get_required_arg(args, 'embedding_size')

    # call process
    def s_call(inf, outf):
        call_args = [
                '-train',     inf,
                '-output',    outf,
                '-cbow',      '1', 
                '-size',      str(embedding_size),
                '-window',    '8',
                '-binary',    '0',
                '-iter',      '15',
                '-min-count', '1',
        ]
        subprocess.call([e] + call_args)

    # create embeddings
    log("Creating input embedding...")
    s_call(infile_src, outfile_src)
    args.embedding_src = outfile_src

    log("Creating output embedding...")
    if infile_src != infile_dst:
        s_call(infile_dst, outfile_dst)
        args.embedding_dst = outfile_dst
    else:
        args.embedding_dst = outfile_src

    log("Done creating embeddings")


def h_compile(cache, args):
    # TODO: automatic dimension reading

    layer_size = get_required_arg(args, 'embedding_size')
    layer_count = get_required_arg(args, 'layer_count')
    maxlen = get_required_arg(args, 'maxlen')
    compile_train = bool(eval(get_required_arg(args, 'compile_train')))

    start_token=None 
    loss='mean_squared_error'
    optimizer=get_required_arg(args, 'optimizer')

    # build model
    log("Building model...")
    model = helpers.build_model(layer_size, layer_count, maxlen,
            start_token, loss, optimizer, compile_train)
            
    outfile = args.output_compiled_model

    if outfile is not None:
        log("Exporting compiled model to {0}".format(outfile))
        helpers.export_model(model, outfile)

    cache['compiled_model'] = model


def h_train(cache, args):
    # load embeddings
    embedding_src = get_embedding(cache, args, 'embedding_src')
    embedding_dst = get_embedding(cache, args, 'embedding_dst')

    # load train dataset
    train_src = get_required_arg(args, 'train_src')
    train_dst = get_required_arg(args, 'train_dst')
    maxlen = get_required_arg(args, 'maxlen')

    X_train, Y_train, M_train, maxlen = helpers.load_dataset(
            embedding_src, embedding_dst,
            train_src, train_dst,
            maxlen)

    # load validation dataset
    validation_src = get_required_arg(args, 'validation_src')
    validation_dst = get_required_arg(args, 'validation_dst')
    maxlen = get_required_arg(args, 'maxlen')

    X_validation, Y_validation, M_validation, maxlen = helpers.load_dataset(
            embedding_src, embedding_dst,
            validation_src, validation_dst,
            maxlen)

    # load model
    modelfile = args.compiled_model
    if modelfile is not None:
        model = helpers.import_model(modelfile)
    elif 'compiled_model' in cache:
        model = cache['compiled_model']
    else:
        print "Error in train: no compiled model provided: exiting"
        exit()

    # load weights (if applicable)
    input_weights = args.model_weights
    if input_weights is not None:
        weights = helpers.import_weights(input_weights)
        model.set_weights(weights)

    # train model
    batch_size = get_required_arg(args, 'batch_size')
    validation_skip = get_required_arg(args, 'validation_skip')
    snapshot_skip = get_required_arg(args, 'snapshot_skip')
    lr_A = get_required_arg(args, 'lr_encoder')
    lr_B = get_required_arg(args, 'lr_decoder')

    timer_start = None

    def get_elapsed():
        return (datetime.utcnow() - timer_start).total_seconds()

    # policy evaluated at beginning of each epoch:
    # stop training if any thresholds are met
    def continue_training(epoch, callback_result):
        def check_threshold_lt(left, right):
            return left is not None and right is not None and left < right
        def check_threshold_gt(left, right):
            return left is not None and right is not None and left > right
        def dg(l):
            try:
                return l()
            except KeyError:
                return None
            except TypeError:
                return None

        loss      = dg(lambda: callback_result['sets']['train']['loss'])
        val_loss  = dg(lambda: callback_result['sets']['validate']['loss'])
        error     = dg(lambda: 1 - callback_result['sets']['train']['summary']['normal, L2']['avg_correct_pct'])
        val_error = dg(lambda: 1 - callback_result['sets']['validate']['summary']['normal, L2']['avg_correct_pct'])

        return not (
            check_threshold_gt(epoch, args.epochs) or \
            check_threshold_gt(get_elapsed(), args.seconds) or \
            (check_threshold_lt(loss, args.loss) and check_threshold_lt(val_loss, args.loss)) or \
            (check_threshold_lt(error, args.error) and check_threshold_lt(val_error, args.error))
        )

    # end of epoch callback: output stats, take snapshot
    snapshot_prefix = args.output_snapshot_prefix
    def epoch_callback(sets, epoch):
        elapsed = get_elapsed()

        log("Begin epoch callback for epoch {0}".format(epoch))
        if validation_skip > 0 and (epoch + 1) % validation_skip == 0:
            from utils import LN
            L1 = lambda vec, x: -LN(vec, x, n=1)

            #w_raw = helpers.generate_D_L1_Usage(
            #            embedding_src, embedding_dst, model, sets[0][1], sets[0][2])
            #DLs = [('normal, L2', None, None),
            #       ('D_L1_Usage, L2', w_raw, None),
            #       ('normal, L1', None, L1),
            #       ('D_L1_Usage, L1', w_raw, L1),
            #]

            DLs = [('normal, L2', None, None),
                   ('normal, L1', None, L1),
            ]

            s = output_dumps.setacc(embedding_src, embedding_dst, model, sets,
                    epoch, elapsed, min(100, len(X_validation)), DLs)
        else:
            set_dicts = {name: {'loss': loss} for name, X, Y, loss in sets}
            s = {'epoch': epoch, 'time': elapsed, 'sets': set_dicts}

        stat.info("{0},".format(s))

        # take snapshot
        if snapshot_prefix and snapshot_skip > 0 and (epoch + 1) % snapshot_skip == 0:
            snapshot_name = "{0}.{1}".format(snapshot_prefix, epoch)
            log("Exporting snapshot weights for epoch {0} to {1}".format(epoch, snapshot_name))
            helpers.export_weights(model, snapshot_name)
            log("Exported snapshot weights for epoch {0}".format(epoch))

        log("End epoch callback for epoch {0}".format(epoch))
        return s

    log("Training model...")
    timer_start = datetime.utcnow()
    model.fit(
            X_train, Y_train, M_train,
            X_validation, Y_validation, M_validation,
            lr_A=lr_A, lr_B=lr_B,
            batch_size=batch_size,
            verbose=1, shuffle=True,
            continue_training=continue_training,
            epoch_callback=epoch_callback
    )

    output_model = args.output_fitted_model
    if output_model is not None:
        log("Exporting fitted model to {0}".format(output_model))
        helpers.export_model(model, output_model)

    output_weights = args.output_model_weights
    if output_weights is not None:
        log("Exporting fitted weights to {0}".format(output_weights))
        helpers.export_weights(model, output_weights)

    cache['fitted_model'] = model


def h_test(cache, args):
    # load model
    model = get_fitted_model(cache, args)

    # load embeddings
    embedding_src = get_embedding(cache, args, 'embedding_src')
    embedding_dst = get_embedding(cache, args, 'embedding_dst')

    # load dataset
    test_src = get_required_arg(args, 'test_src')
    test_dst = get_required_arg(args, 'test_dst')

    maxlen = model.model_B.steps

    X, Y, M, maxlen = helpers.load_dataset(
            embedding_src, embedding_dst,
            test_src, test_dst,
            maxlen)

    # test
    test_format = args.test_format.lower()
    print test_format


    ## generate DLs from training corpus (todo: make this part of the model)
    #train_src = get_required_arg(args, 'train_src')
    #train_dst = get_required_arg(args, 'train_dst')
    #X_t, Y_t, M_t, maxlen_t = helpers.load_dataset(
    #        embedding_src, embedding_dst,
    #        train_src, train_dst,
    #        maxlen)

    DLs = []
    DLs.append(('normal', None))

    #print 'generate D_L1_Usage'
    #D_L1_Usage = helpers.generate_D_L1_Usage(embedding_src, embedding_dst, model, X_t, Y_t)
    #print 'D_L1_Usage', D_L1_Usage
    #DLs.append(('D_L1_Usage', D_L1_Usage))

    print 'done DLs'

    output_dumps.nptest_dict(embedding_src, embedding_dst, model, X, Y, M, DLs)

    for DL in DLs:
        print DL

def h_export(cache, args):
    # load model
    model = get_fitted_model(cache, args)

    # get and export weights
    output_weights = get_required_arg(args, 'output_model_weights')
    print "Exporting fitted weights to {0}".format(output_weights)
    helpers.export_weights(model, output_weights)


def h_interactive(cache, args):
    # load model
    model = get_fitted_model(cache, args)

    # load embeddings
    embedding_src = get_embedding(cache, args, 'embedding_src')
    embedding_dst = get_embedding(cache, args, 'embedding_dst')

    maxlen = model.model_B.steps

    D = None

    print "Translate:"
    while True:
        raw = raw_input("> ")
        raw_eof = raw.replace('.', ' .').replace('!', ' !').replace('?', ' ?')
        tokens = filter(lambda x: x != "", [x.strip() for x in raw_eof.split(' ')])
        fit_tokens = tokens[:min(len(tokens), maxlen)]
        print fit_tokens
        X = embedding_src.convert_sentence(fit_tokens, reverse=True)
        R = model.predict_batch([X], batch_size=1)[0]
        R_t = [embedding_dst.matchN(o, 6, w=D) for o in R] 
        R_c = embedding_dst.clip([x[0][0] for x in R_t])
        print R_c
        print ' '.join(R_c)
        print '=========================='
        for r in R_t[:len(R_c)]:
            s = ''
            for item in r:
                s += '{0}: {1}; '.format(item[0], item[2])
            print s
        print ""
        
main()    

