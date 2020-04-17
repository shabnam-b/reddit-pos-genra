import argparse
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import CharacterEmbeddings, FlairEmbeddings, TokenEmbeddings, StackedEmbeddings
from typing import List
from flair.embeddings import FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-T', '--train', help='Path to train doc',required=True)
args.add_argument('-t', '--test', help='Path to test doc',required=True)
args.add_argument('-d', '--dev', help='Path to dev doc',required=True)
args.add_argument('-r', '--results', help='Path to the results file',required=True)
args.add_argument('-m', '--model', help='Path to the model directory',required=True)
args = args.parse_args()


def train():
    columns = {0: 'text', 1: 'pos'}
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus('', columns,
                                  train_file=args.train,
                                  test_file=args.test,
                                  dev_file=args.dev)

    tag_dictionary = corpus.make_tag_dictionary(tag_type='pos')

    # initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        CharacterEmbeddings(),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='pos',
                                            use_crf=True)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(args.model,
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=150)


def predict():
    model = SequenceTagger.load(args.model + '/final-model.pt')
    sentences = []
    with open(args.test) as f:
        s = ''
        for line in f:
            if line == '\n' and len(s) > 0:
                sentences.append(s)
                s = ''
            else:
                s += line.split('\t')[0] + ' '
    sents = [(len(s.split()), i, s) for i, s in enumerate(sentences)]
    sents.sort(key=lambda x: x[0], reverse=True)
    sentences = [s[2] for s in sents]
    preds = model.predict(sentences)

    # sort back
    sents = [tuple(list(sents[i]) + [s]) for i, s in enumerate(preds)]
    sents.sort(key=lambda x: x[1])
    sents = [s[3] for s in sents]

    output = open(args.results, 'w')
    for s in sents:
        for tok in s.tokens:
            output.write(tok.tags['pos'].value + '\n')


if __name__ == "__main__":
    print("Starting loading data and training")
    train()
    predict()
