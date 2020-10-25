#encoding:utf-8
import tensorflow as tf
from gensim import corpora, models, similarities
import numpy as np
import math
import fasttext
import os, argparse, time, random, re
from BiLSTM import BiLSTM_CRF
from utils import str2bool, get_logger, getlnglat, get_date_info
from entity import get_entity
from corpus import read_corpus, read_train, vocab_build, read_dictionary, tag2label, random_embedding

## TF Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

## hyperparameters
parser = argparse.ArgumentParser(description='Similarity!')
parser.add_argument('--data', type=str, default='data', help='path of data')

#for BiLSTM_CRF
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')

# actions
parser.add_argument('--mode', type=str, default='ner_demo', help='ner_gendict/ner_train/ner_test/ner_demo/gensim_tags/gensim_predict')
parser.add_argument('--model', type=str, default='1603464530', help='model for test and demo')
args = parser.parse_args()

#corpus
corpus_root = os.path.join('.', args.data)
save_root = os.path.join('.', 'save')

classifier_corpus_path = os.path.join(corpus_root, 'classifier_corpus.txt')
classifier_train_path = os.path.join(corpus_root, 'classifier_train.txt')
classifier_model_path = os.path.join(save_root, 'classifier_model')

ner_corpus_path = os.path.join(corpus_root, 'ner_corpus.txt')
ner_train_path = os.path.join(corpus_root, 'ner_train.txt')
ner_result_path = os.path.join(save_root, 'ner_result.txt')
ner_vocab_path = os.path.join(save_root, 'word2id.pkl')

## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.model

save_path = os.path.join(save_root, timestamp)
if not os.path.exists(save_path): os.makedirs(save_path)

summary_path = os.path.join(save_path, "summaries")
paths['summary_path'] = summary_path

if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(save_path, "checkpoints/")

if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix

result_path = os.path.join(save_path, "results")
paths['result_path'] = result_path

if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## generate dictionary for BiLSTM-CRF
def gen_dictionary():
    gen_vocab = vocab_build(ner_vocab_path, ner_train_path, min_count=1)

## get char embeddings
def get_embedding():
    if args.pretrain_embedding == 'random':
        embeddings = random_embedding(word2id, args.embedding_dim)
    else:
        embedding_path = 'pretrain_embedding.npy'
        embeddings = np.array(np.load(embedding_path), dtype='float32')

    return embeddings

## get char embeddings
def ner_train():
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ner_train_data = read_train(ner_train_path)
    data_size = len(ner_train_data)

    split = math.floor(data_size*0.9)

    train_data = ner_train_data[:split]
    test_data = ner_train_data[split:]

    # hyperparameters-tuning, split train/test
    train_size = len(train_data)
    test_size = len(test_data)

    print("train data: {0}\ntest data: {1}".format(train_size, test_size))
    model.train(train=train_data, test=test_data)

    ## train model on the whole training data
    ## print("train data: {}".format(len(train_data)))
    ## model.train(train=train_data, test=test_data)  # use test_data as the test_data to see overfitting phenomena

## testing model
def ner_test():
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ner_train_data = read_train(ner_train_path)
    data_size = len(ner_train_data)
    split = math.floor(data_size*0.9)

    test_data = ner_train_data[split:]
    test_size = len(test_data)

    print("test data: {}".format(test_size))
    model.test(test_data)

## get ner tag
def one_ner_tag(sent):
    ckpt_file = tf.train.latest_checkpoint(model_path)
    #print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()

    classifier = fasttext.load_model(classifier_model_path)

    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt_file)
        if sent == '' or sent.isspace():
            print('empty sentence!')
            return None

        sent = list(sent.replace(' ','',10).strip())
        sent_data = [(sent, ['O'] * len(sent))]
        tags = model.get_ner_tag(sess, sent_data)

        Location, Time, Means, Thing = get_entity(tags, sent)
        print('Location: {}\nTime: {}\nMeans: {}\nThing: {}'.format(Location, Time, Means, Thing))

        words = []
        if len(Time)>0:
            time_info = get_date_info(Time[0])
            print(time_info)
            if time_info is None and len(Time)>1:
                time_info = get_date_info(Time[1])
            print(time_info)
            words += list(time_info)
        else:
            print('NoTime {}'.format(sent))
            words.append('NoTime')

        if len(Location)>0:
            location_info = getlnglat(Location[0])
            print(location_info)
            words.append(location_info)
        else:
            print('NoLocation {}'.format(sent))
            words.append('NoLocation')

        if len(Means)>0:
            words += Means
        else:
            print('NoMeans {}'.format(sent))
            words.append('NoMeans')

        if len(Thing)>0:
            category = classifier.predict(Thing, k=1)[0][0][0].replace('__label__','')
            print(category)
            words.append(category)
        else:
            print('NoClass {}'.format(sent))
            words.append('NoClass')

        return words

## get tags for corpus
def gensim_tags():
    ckpt_file = tf.train.latest_checkpoint(model_path)
    #print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()

    classifier = fasttext.load_model(classifier_model_path)

    with tf.Session(config=config) as sess:
        print('============= ner_tags =============')
        saver.restore(sess, ckpt_file)

        ner_result_fb = open(ner_result_path, 'a+')

        corpus_lines = read_corpus(ner_corpus_path)
        for ner_line in corpus_lines:
            ner_line = ner_line.strip()
            ner_line = re.sub(r'\s+', '', ner_line)
            ner_line = re.sub(r'日(凌晨|早晨|上午|中午|下午|晚上|深夜)+', '日', ner_line)
            line_data = list(ner_line.replace(' ','',10).strip())
            line_data = [(line_data, ['O'] * len(line_data))]
            tags = model.get_ner_tag(sess, line_data)
            Location, Time, Means, Thing = get_entity(tags, ner_line)
            print('Location: {}\nTime: {}\nMeans: {}\nThing: {}'.format(Location, Time, Means, Thing))

            words = []
            if len(Time)>0:
                time_info = get_date_info(Time[0])
                print(time_info)
                if time_info is None and len(Time)>1:
                    time_info = get_date_info(Time[1])
                print(time_info)
                words += list(time_info)
            else:
                print('NoTime {}'.format(ner_line))
                words.append('NoTime')

            if len(Location)>0:
                location_info = getlnglat(Location[0])
                print(location_info)
                words.append(location_info)
            else:
                print('NoLocation {}'.format(ner_line))
                words.append('NoLocation')

            if len(Means)>0:
                words += Means
            else:
                print('NoMeans {}'.format(ner_line))
                words.append('NoMeans')

            if len(Thing)>0:
                category = classifier.predict(Thing, k=1)[0][0][0].replace('__label__','')
                print(category)
                words.append(category)
            else:
                print('NoClass {}'.format(ner_line))
                words.append('NoClass')

            print(words)
            ner_result_fb.write(' '.join(words)+"\t"+ner_line+"\n")

        ner_result_fb.close()

## demo
def ner_demo():
    ckpt_file = tf.train.latest_checkpoint(model_path)
    #print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tags = model.get_ner_tag(sess, demo_data)
                print(tags)
                Location, Time, Means, Thing = get_entity(tags, demo_sent)
                print('Location: {}\nTime: {}\nMeans: {}\nThing: {}'.format(Location, Time, Means, Thing))

                if len(Time)>0:
                    print(get_date_info(Time[0]))
                if len(Location)>0:
                    print(getlnglat(Location[0]))

def similarity(sent, topN=10):

    corpus_lines = read_corpus(ner_result_path)
    texts = [line.split("\t")[0].split(' ') for line in corpus_lines]

    keywords = one_ner_tag(sent)

    dictionary = corpora.Dictionary(texts)
    num_features = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)

    new_vec = dictionary.doc2bow(keywords)
    # 相似度计算
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features)
    # index = similarities.Similarity('-Similarity-index', corpus, num_features)
    # print('\nTF-IDF模型的稀疏向量集：')
    # for i in tfidf[corpus]:
    #     print(i)
    # print('\nTF-IDF模型的keyword稀疏向量：')
    # print(tfidf[new_vec])

    sims = index[tfidf[new_vec]]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    print("\n相似度计算")
    print('Words: {}\nText: {}\n'.format(keywords, sent))

    for k,v in sims[:topN]:
            i = int(k)
            print('Similarity: {}\nWords: {}\nText: {}'.format(v, corpus_lines[i].split("\t")[0].split(' '), corpus_lines[i].split("\t")[1]))



word2id = read_dictionary(ner_vocab_path)
embeddings = get_embedding()

if args.mode == 'ner_gendict':
    gen_dictionary()

if args.mode == 'ner_train':
    ner_train()

if args.mode == 'ner_test':
    ner_test()

if args.mode == 'ner_demo':
    ner_demo()

if args.mode == 'gensim_tags':
    gensim_tags()

if args.mode == 'gensim_predict':
    while(1):
        print('Please input your sentence:')
        sent = input()
        if sent == '' or sent.isspace():
            print('See you next time!')
            break
        else:
            sent = sent.strip()
            similarity(sent)

