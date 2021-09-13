import argparse
from exeval import sequence_labeling, snli, subjectivity, relation_extraction, sentence_sentiment, document_sentiment
import logging
import os
import json
import sys


MODULES = {
    'sequence_labeling': sequence_labeling,
    'snli': snli,
    'subjective': subjectivity,
    'relation': relation_extraction,
    'sentence_sentiment': sentence_sentiment,
    'document_sentiment': document_sentiment,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--backend', type=str)
    parser.add_argument('--vector_path',
                        required=True,
                        help='path to vectors (in text format)')


    subparser=parser.add_subparsers(help='task to run', dest='task')

    for name, module in MODULES.items():
        sp = subparser.add_parser(name)
        module.mk_parser(sp)

    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.DEBUG)

    if args.backend:
        os.environ['KERAS_BACKEND'] = args.backend

    logging.info('Running task {} with vectors {}'.format(args.task, args.vector_path))
    metrics = args.go(args)
    args = vars(args)

    # Remove "go" from args because it is only there for logistical reasons.
    del args['go']
    # Remove "log" from args because it is not relevant for the output.
    del args['log']

    output = {
        'parameters': args,
        'metrics': metrics,
    }

    json.dump(output, sys.stdout)
    sys.stdout.write('\n')










