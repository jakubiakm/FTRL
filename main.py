from ftrl import FTRL
from csv import DictReader
from datetime import datetime
from math import log
import argparse

def main(args):
    model = FTRL(args.alpha, args.beta, args.L1, args.L2)
    train(model)
    test(model)

def load_data_enumarator(path):
    for _, row in enumerate(DictReader(open(path), delimiter=',')):
        clicked = 0
        if 'target' in row:
            if row['target'] == '1':
                #kliknięcię wystąpiło
                clicked = 1.
        # budowa wektora cech za pomocą hash trick
        feature = []
        for key in row:
            if key != 'target' and key != 'ID':
                feature.append(abs(hash(key + '_' + row[key])) % 2 ** 28)
        yield feature, clicked

def train(ftrl_model):
    epoch = 10
    print('alpha:', ftrl_model.alpha, 'beta:', ftrl_model.beta, 'L1:', ftrl_model.L1, 'L2:', ftrl_model.L2)
    print('epoch, count, logloss')
    for epoch_number in range(epoch):
        loss = 0
        count = 0
        train_data = load_data_enumarator('data/train.csv')
        for feature, clicked in train_data:  # data is a generator
            probability = ftrl_model.predict(feature)
            loss += logloss(probability, clicked)
            ftrl_model.update_model(feature, probability, clicked)
            count += 1
            if count % 1000 == 0:
                print(epoch_number + 1, count, loss/count)

def test(ftrl_model):
    test_data = load_data_enumarator('data/test.csv')
    print ('write result')
    print('ID,target\n')
    for feature, _ in test_data:
        probability = ftrl_model.predict(feature)
        print(probability)

def logloss(p, y):
    return -log(p) if y == 1. else -log(1. - p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FTRL')
    parser.add_argument('--alpha', type=float, required=True, default=0.005)
    parser.add_argument('--beta', type=float, required=True, default=1)
    parser.add_argument('--L1', type=float, required=True, default=0)
    parser.add_argument('--L2', type=float, required=True, default=1)

    args = parser.parse_args()
    main(args)