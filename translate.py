import sys
import argparse
import subprocess
import collections

import helpers
import output_dumps

from log import log

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
    parser = argparse.ArgumentParser(description="LSTM Encoder Decoder")
    
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
    parser.add_argument('--test-src', dest='test_src', type=str,
            help="Test sentences for source (encoder) network")
    parser.add_argument('--test-dst', dest='test_dst', type=str,
            help="Test sentences for destination (decoder) network")
    parser.add_argument('--test-both', dest='test_both', type=str,
            help="Test sentences for both encoder and decoder network")

    # parameters
    parser.add_argument('--embedding-size', dest='embedding_size', type=int,
            help="Embedding vector size")
    parser.add_argument('--layers', dest='layer_count', type=int,
            help="Network layer count")
    parser.add_argument('--max-sentence-length', dest='maxlen', type=int,
            help="Maximum sentence length")
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            default=16,
            help="Training batch size")
    parser.add_argument('--epochs', dest='epochs', type=int,
            help="Number of training epochs")
    parser.add_argument('--validation-pct', dest='validation_pct', type=float,
            default=0,
            help="Percent of training set used as validation set")
    parser.add_argument('--validation-skip', dest='validation_skip', type=float,
            default=10,
            help="Amount of epochs to skip before outputting validation translations")
    parser.add_argument('--lr-encoder', dest='lr_encoder', type=float,
            default=None,
            help="Learning rate for encoder")
    parser.add_argument('--lr-decoder', dest='lr_decoder', type=float,
            default=None,
            help="Learning rate for decoder")
    parser.add_argument('--lr-both', dest='lr_both', type=float,
            default=None,
            help="Learning rate for both")
    parser.add_argument('--format', dest='test_format', type=str,
            default='',
            help="Test output format (options: pairs (default), simple, complex)")
    parser.add_argument('--compile-train', dest='compile_train', type=str,
            default='True',
            help="Compile training functions for model")

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

    
    args = parser.parse_args()

    # handle 'both' arguments here
    if args.train_both is not None:
        args.train_src = args.train_both
        args.train_dst = args.train_both
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
    model = None
    if input_model is not None:
        model = helpers.import_model(input_model)
    elif input_weights is not None:
        model = cache['compiled_model']
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
                '-train',  inf,
                '-output', outf,
                '-cbow',   '1', 
                '-size',   str(embedding_size),
                '-window', '8',
                '-binary', '0',
                '-iter',   '15',
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
    optimizer='rmsprop'
    #TODO: add hyperparams

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

    # load dataset
    train_src = get_required_arg(args, 'train_src')
    train_dst = get_required_arg(args, 'train_dst')
    maxlen = get_required_arg(args, 'maxlen')

    X, Y, M, maxlen = helpers.load_dataset(
            embedding_src, embedding_dst,
            train_src, train_dst,
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
    epochs = get_required_arg(args, 'epochs')
    validation_split = get_required_arg(args, 'validation_pct')
    validation_skip = get_required_arg(args, 'validation_skip')
    lr_A = get_required_arg(args, 'lr_encoder')
    lr_B = get_required_arg(args, 'lr_decoder')

    # validation function
    def vf(sets, epoch_no):
        from utils import LN
        w_raw = helpers.generate_D_L1_Usage(
                    embedding_src, embedding_dst, model, sets[0][1], sets[0][2])
        L1 = lambda vec, x: -LN(vec, x, n=1)
        DLs = [('normal, L2', None, None),
               ('D_L1_Usage, L2', w, None),
               ('normal, L1', None, L1),
               ('D_L1_Usage, L1', w, L1),
        ]
        #d = output_dumps.datadump(embedding_src, embedding_dst, model, sets, epoch_no, 25, 6, DLs)
        #print d

        s = output_dumps.setacc(embedding_src, embedding_dst, model, sets, epoch_no, None, DLs)
        print s

    print "Training model..."
    model.fit(X, Y, M,
            lr_A=lr_A, lr_B=lr_B,
            batch_size=batch_size, nb_epoch=epochs, verbose=1,
            validation_split=validation_split, validation_skip=validation_skip,
            validation_callback=vf,
    )

    output_model = args.output_fitted_model
    if output_model is not None:
        print "Exporting fitted model to {0}".format(output_model)
        helpers.export_model(model, output_model)

    output_weights = args.output_model_weights
    if output_weights is not None:
        print "Exporting fitted weights to {0}".format(output_weights)
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


    # generate DLs from training corpus (todo: make this part of the model)
    train_src = get_required_arg(args, 'train_src')
    train_dst = get_required_arg(args, 'train_dst')
    X_t, Y_t, M_t, maxlen_t = helpers.load_dataset(
            embedding_src, embedding_dst,
            train_src, train_dst,
            maxlen)

    DLs = []
    DLs.append(('normal', None))

    print 'generate D_L1_Usage'
    D_L1_Usage = helpers.generate_D_L1_Usage(embedding_src, embedding_dst, model, X_t, Y_t)
    print 'D_L1_Usage', D_L1_Usage
    DLs.append(('D_L1_Usage', D_L1_Usage))

    print 'done DLs'

    output_dumps.nptest_dict(embedding_src, embedding_dst, model, X, Y, DLs)

    for DL in DLs:
        print DL

    #if test_format == 'complex':
    #    output_dumps.realtest(embedding_src, embedding_dst, model, X, Y)
    #elif test_format == 'simple':
    #    output_dumps.simpletest(embedding_src, embedding_dst, model, X, Y)
    #else:
    #    output_dumps.nptest(embedding_src, embedding_dst, model, X, Y)


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

    #D = {'x': 1.2917847775734186, 'ONE': 1.8946203078645807, 'TWO': 1.9620620642372502, 'THREE': 1.928626529470113, 'DONE': 1.8997229195961116, 'FOUR': 1.9709916347674294}
    #D = {'a':1, '.':1, 'A':1, 'AA':1, 'AAA':1, 'AAAA':1}

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


