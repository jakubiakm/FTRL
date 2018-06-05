from ftrl import FTRL
from csv import DictReader
from datetime import datetime
from math import log

def main():
    model = FTRL(0.005, 1, 0, 1)
    train(model)
    test(model)

def load_data_enumarator(path):
    for t, row in enumerate(DictReader(open(path), delimiter=',')):
        try:
            ID = row['ID']
            del row['ID']
        except:
            pass 
        y = 0.
        target='target'
        if target in row:
            if row[target] == '1':
                #kliknięcię wystąpiło
                y = 1.
            del row[target]
        # budowany jest wektor cech za pomocą hash trick
        x = []
        for key in row:
            value = row[key]
            index = abs(hash(key + '_' + value)) % 2 ** 28   
            x.append(index)
        yield x, y

def train(ftrl_model):
    epoch = 10
    print('alpha:', ftrl_model.alpha, 'beta:', ftrl_model.beta, 'L1:', ftrl_model.L1, 'L2:', ftrl_model.L2)
    print('epoch, count, logloss')
    for e in range(epoch):
        loss = 0
        count = 0
        train_data = load_data_enumarator('data/train.csv')
        for x, y in train_data:  # data is a generator
            p = ftrl_model.predict(x)
            loss += logloss(p, y)
            ftrl_model.update(x, p, y)
            count+=1
            if count%1000==0:
                print(e + 1, count, loss/count)

def test(ftrl_model):
    test_data = load_data_enumarator('data/test.csv')
    print ('write result')
    print('ID,target\n')
    for x, y in test_data:
        p = ftrl_model.predict(x)
        print(p)

def logloss(p, y):
    return -log(p) if y == 1. else -log(1. - p)

if __name__ == "__main__":
    main()