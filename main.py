import xgboost
import os
import argparse

#   general arguments
parser = argparse.ArgumentParser(description='use xgboost to train a text classifier and do a predict')
parser.add_argument('-fsave-model', type=str, default='./models/new%d.model')
parser.add_argument('-ftrain-svm', type=str, default='./data/train.svm.txt')
parser.add_argument('-ftest-svm', type=str, default='./data/test.svm.txt')
parser.add_argument('-fval-svm', type=str, default='./data/val.svm.txt')
parser.add_argument('-fpred-out', type=str, default='./predictions/new%d.csv')
parser.add_argument('-ftest-csv', type=str, default='./data/sample.csv')
parser.add_argument('-auto', action='store_true', default=False)
parser.add_argument('-cv', action='store_true', default=False)
parser.add_argument('-acc', action='store_true', default=False)
parser.add_argument('-predict-only', action='store_true', default=False)
parser.add_argument('-d-features', type=int, default=25)
#   cross validataion arguments
# parser.add_argument('-nfold', type=int, default=4)

#   train arguments
parser.add_argument('-snapshot', type=str, default=None)
parser.add_argument('-num-round', type=int, default=5000)
parser.add_argument('-max-depth', type=int, default=8)
parser.add_argument('-subsample', type=float, default=1)
#parser.add_argument('-early-stopping-rounds', type=int, default=3)
parser.add_argument('-colsample-bytree', type=float, default=1)
parser.add_argument('-colsample-bylevel', type=float, default=1)
parser.add_argument('-eta', type=float, default=0.2)            #   step size shrinkage used in update to prevents overfitting
#parser.add_argument('-max-leaf-nodes', type=int, default=80)
parser.add_argument('-scale-pos-weight', type=float, default=12.0)
parser.add_argument('-silent', type=int, default=0)
parser.add_argument('-objective', type=str, default='reg:squarederror')
parser.add_argument('-nthread', type=int, default=4)
parser.add_argument('-eval-metric', type=str, default='error')

args = parser.parse_args()

def polarity(x):
    if x >= 0.5:
        return 1
    else:
        return -1

def get_label(x):
    if x >= 0.5:
        return 1
    else:
        return 0

def get_acc(bst):
    ypred = bst.predict(D_val)
    labels = D_val.get_label()
    cnt = 0
    for idx in range(len(ypred)):
        t = get_label(ypred[idx])
        if t == labels[idx]:
            cnt += 1
    print('acc is %f' % ( cnt / len(ypred)))

def predict(bst):
    print('start predict...')
    ypred = bst.predict(D_test)
    cur = 0
    while os.path.exists(args.fpred_out % cur):
        cur += 1

    ftest_csv = open(args.ftest_csv, "r")
    fpred_out = open(args.fpred_out % cur, "w")

    for idx, line in enumerate(ftest_csv.readlines(), -1):
        if idx == -1:
            fpred_out.write("%s" % line)
        else:
            fpred_out.write('%s,%f\n' % (line.split(",")[0], ypred[idx]))
    ftest_csv.close()
    fpred_out.close()

if __name__ == '__main__':
    D_test = xgboost.DMatrix(args.ftest_svm)
    D_train = xgboost.DMatrix(args.ftrain_svm)
    D_val = xgboost.DMatrix(args.fval_svm)

    if args.acc == True:
        bst = xgboost.Booster(model_file=os.path.join('models', args.snapshot))
        get_acc(bst)
        exit(0)

    if args.predict_only == True:
        if args.snapshot == None:
            exit(0)
        else:
            bst = xgboost.Booster(model_file=os.path.join('models',args.snapshot))
            predict(bst)
            exit(0)

    print('train model...')
    eval_list = [(D_train, 'train'), (D_val, 'valid')]
    if args.snapshot == None:
        bst = xgboost.train(args.__dict__, D_train, args.num_round, eval_list)
    else:
        print('continue from snapshot %s' % args.snapshot)
        bst = xgboost.train(args.__dict__, D_train, args.num_round, eval_list, xgb_model=os.path.join('models', args.snapshot))

    print('save model...')
    cur = 0
    while os.path.exists(args.fsave_model % cur):
        cur += 1
    fname = args.fsave_model % cur
    bst.save_model(fname)
    print('%s saved.' % (fname))

