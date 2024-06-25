
import os
import pdb
import random
import math
import warnings
import argparse

import torch
from torchvision import transforms, datasets, models
import torch.utils.data

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

import sys

warnings.filterwarnings("ignore")


from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.ClassifyCount import ClassifyCount
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.ProbabilisticClassifyCount import ProbabilisticClassifyCount
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.AdjustedClassifyCount import AdjustedClassifyCount
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.ProbabilisticAdjustedClassifyCount import ProbabilisticAdjustedClassifyCount
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.X import X
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.MAX import MAX
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.MedianSweep import MedianSweep
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.MS2 import MedianSweep2
from QuantifiersLibrary.quantifiers.ClassifyCountCorrect.T50 import T50

from QuantifiersLibrary.quantifiers.DistributionMatching.DyS import DyS
from QuantifiersLibrary.quantifiers.DistributionMatching.HDy import HDy
from QuantifiersLibrary.quantifiers.DistributionMatching.SORD import SORD
from QuantifiersLibrary.quantifiers.DistributionMatching.FM import FM
from QuantifiersLibrary.quantifiers.DistributionMatching.FMM import FMM
from QuantifiersLibrary.quantifiers.DistributionMatching.SMM import SMM
from QuantifiersLibrary.quantifiers.DistributionMatching.EMQ import EMQ

from QuantifiersLibrary.utils import Quantifier_Utils


def transform_images(image_folder):
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    images_transformed = datasets.ImageFolder(os.path.join(image_folder), transform=data_transform)

    return images_transformed

def setting_model(cnn_model_name, pt):
    if cnn_model_name == 'resnet50':
        cnn_model = models.resnet50()
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device) # Device from main (GPU or CPU)

        num_features = cnn_model.module.fc.in_features
        cnn_model.module.fc = torch.nn.Linear(num_features, 2) # 2 because it's binary classification, change it to get the numbers of folder maybe!

    elif cnn_model_name == 'vgg16':
        cnn_model = models.vgg16()
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)

        num_features = cnn_model.module.classifier[6].in_features
        cnn_model.module.classifier[6] = torch.nn.Linear(num_features, 2)

    elif cnn_model_name == 'googlenet':
        if pt == 1:
            cnn_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        else:
            print('npt')
            cnn_model = models.googlenet()
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)

        num_features = cnn_model.module.fc.in_features
        cnn_model.module.fc = torch.nn.Linear(num_features, 2)
    else:
        print('Invalid model')
        sys.exit(1)

    cnn_model = cnn_model.cuda()

    return cnn_model

def run_quantifier(quant, test_scores, tprfpr, validation_scores, thr=0.5, measure='topsoe'):

    proportion_result = None

    if quant == 'CC':
        cc = ClassifyCount(classifier=None, threshold=thr)
        proportion_result = cc.get_class_proportion(scores=test_scores)

    elif quant == 'PCC':
        pcc = ProbabilisticClassifyCount(classifier=None)
        proportion_result = pcc.get_class_proportion(scores=test_scores)

    elif quant == 'ACC':
        acc = AdjustedClassifyCount(classifier=None, threshold=thr)
        acc.tprfpr = tprfpr
        proportion_result = acc.get_class_proportion(scores=test_scores)

    elif quant == 'PACC':
        pacc = ProbabilisticAdjustedClassifyCount(classifier=None, threshold=thr)
        pacc.tprfpr = tprfpr
        proportion_result = pacc.get_class_proportion(scores=test_scores)

    elif quant == 'X':
        x = X(classifier=None)
        x.tprfpr = tprfpr
        proportion_result = x.get_class_proportion(scores=test_scores)

    elif quant == 'MAX':
        quant_max = MAX(classifier=None)
        quant_max.tprfpr = tprfpr
        proportion_result = quant_max.get_class_proportion(scores=test_scores)

    elif quant == 'MS':
        ms = MedianSweep(classifier=None)
        ms.tprfpr = tprfpr
        proportion_result = ms.get_class_proportion(scores=test_scores)

    elif quant == 'MS2':
        ms2 = MedianSweep2(classifier=None)
        ms2.tprfpr = tprfpr
        proportion_result = ms2.get_class_proportion(scores=test_scores)

    elif quant == 'T50':
        t50 = T50(classifier=None)
        t50.tprfpr = tprfpr
        proportion_result = t50.get_class_proportion(scores=test_scores)

    elif quant == 'DyS':
        dys = DyS(classifier=None, similarity_measure=measure, data_split=None)

        dys.p_scores = validation_scores[validation_scores['class'] == 1]
        dys.p_scores = dys.p_scores['score'].tolist()

        dys.n_scores = validation_scores[validation_scores['class'] == 0]
        dys.n_scores = dys.n_scores['score'].tolist()

        dys.test_scores = [score[1] for score in test_scores]

        proportion_result = dys.get_class_proportion()

    elif quant == 'HDy':
        hdy = HDy(classifier=None, data_split=None)

        hdy.p_scores = validation_scores[validation_scores['class'] == 1]
        hdy.p_scores = hdy.p_scores['score'].tolist()

        hdy.n_scores = validation_scores[validation_scores['class'] == 0]
        hdy.n_scores = hdy.n_scores['score'].tolist()

        hdy.test_scores = [score[1] for score in test_scores]

        proportion_result = hdy.get_class_proportion()

    elif quant == 'SORD':
        sord = SORD(classifier=None, data_split=None)

        sord.p_scores = validation_scores[validation_scores['class'] == 1]
        sord.p_scores = sord.p_scores['score'].tolist()

        sord.n_scores = validation_scores[validation_scores['class'] == 0]
        sord.n_scores = sord.n_scores['score'].tolist()

        sord.test_scores = [score[1] for score in test_scores]

        proportion_result = sord.get_class_proportion()

    elif quant == 'SMM':
        smm = SMM(classifier=None, data_split=None)

        smm.p_scores = validation_scores[validation_scores['class'] == 1]
        smm.p_scores = smm.p_scores['score'].tolist()

        smm.n_scores = validation_scores[validation_scores['class'] == 0]
        smm.n_scores = smm.n_scores['score'].tolist()

        smm.test_scores = [score[1] for score in test_scores]

        proportion_result = smm.get_class_proportion()

    elif quant == 'FM':
        fm = FM(classifier=None, data_split=None)

        fm.train_scores = [[round(1-score,4), score] for score in validation_scores['score']]
        fm.train_scores = np.array(fm.train_scores)

        fm.test_scores = np.array(test_scores)
        fm.train_labels = validation_scores['class']
        fm.nclasses = 2
        
        proportion_result = fm.get_class_proportion()

    elif quant == 'FMM':
        fmm = FMM(classifier=None, data_split=None)

        fmm.train_scores = validation_scores
        fmm.test_scores = np.array(test_scores)

        fmm.train_labels = validation_scores['class']
        fmm.nclasses = np.unique(fmm.train_labels)

        train_distrib = np.zeros((fmm.nbins * fmm.ndescriptors, len(fmm.nclasses)))

        for n_cls, cls in enumerate(fmm.nclasses):
            descr = 0
            train_distrib[descr * fmm.nbins:(descr + 1) * fmm.nbins, n_cls] = np.histogram(fmm.train_scores[fmm.train_scores['class'] == cls]['score'], bins=fmm.nbins, range=(0., 1.))[0]
            train_distrib[:, n_cls] = train_distrib[:, n_cls] / (fmm.train_scores[fmm.train_scores['class'] == cls].shape[0])

        train_distrib = np.cumsum(train_distrib, axis=0)
        fmm.train_distrib = train_distrib

        proportion_result = fmm.get_class_proportion()

    elif quant == 'EMQ':
        emq = EMQ(classifier=None, data_split=None)

        emq.train_labels = validation_scores['class']
        emq.test_scores = np.array(test_scores)
        emq.nclasses = 2

        proportion_result = emq.get_class_proportion()

    return proportion_result

def get_best_threshold(pos_prop, pos_scores, thr=0.5, tolerance=0.01):
    min = 0.0
    max = 1.0
    max_iteration = math.ceil(math.log2(len(pos_scores))) * 2 + 10
    for _ in range(max_iteration):
        new_proportion = sum(1 for score in pos_scores if score >= thr) / len(pos_scores)

        if abs(new_proportion - pos_prop) < tolerance:
            return thr

        elif new_proportion > pos_prop:
            min = thr
            thr = (thr + max) / 2

        else:
            max = thr
            thr = (thr + min) / 2

    return thr

def new_get_best_threshold(pos_prop, pos_scores, threshold=0.5, tolerance=0.01):
    low = 0.0
    high = 1.0
    max_iterations = math.ceil(math.log2(len(pos_scores))) * 2
    thresholds = {}
    for _ in range(max_iterations):
        positive_proportion = sum(1 for score in pos_scores if score > threshold) / len(
            pos_scores
        )
        error = round(abs(positive_proportion - pos_prop), 3)
        if error not in thresholds:
            thresholds[error] = threshold
        if error < tolerance:
            return threshold
        if positive_proportion > pos_prop:
            low = threshold
            threshold = (threshold + high) / 2
        else:
            high = threshold
            threshold = (threshold + low) / 2
    threshold = thresholds[min(thresholds.keys())]
    return threshold

def classifier_accuracy(pos_proportion, pos_test_scores, labels):
    sorted_scores = sorted(pos_test_scores)

    threshold = get_best_threshold(pos_proportion, sorted_scores)
    threshold2 = new_get_best_threshold(pos_proportion, sorted_scores)

    pred_labels = [1 if score >= threshold else 0 for score in pos_test_scores]
    pred_labels2 = [1 if score >= threshold2 else 0 for score in pos_test_scores]

    corrects = sum(1 for a, b in zip(pred_labels, labels) if a == b)
    accuracy = round(corrects / len(pred_labels), 2)

    corrects2 = sum(1 for a, b in zip(pred_labels2, labels) if a == b)
    accuracy2 = round(corrects2 / len(pred_labels2), 2)

    return accuracy, threshold, accuracy2, threshold2

def run_fscore(pos_probs, label_list, thr):
    pred = [1 if prob >= thr else 0 for prob in pos_probs]
    fscore = f1_score(label_list, pred)
    return fscore

def experiment(model, dts_images, val_scores, model_name, boolean_pt):

    columns = ['sample', 'test_size', 'alpha', 'actual_prop', 'pred_prop', 'abs_error', 'acc', 'f_score', 'thr',
               'acc2', 'thr2', 'quantifier', 'model', 'pt']
    result_table = pd.DataFrame(columns=columns)

    pos_index = [i for i in range(len(dts_images)) if dts_images[i][1] == 1]
    neg_index = [i for i in range(len(dts_images)) if dts_images[i][1] == 0]

    pos_dataset = torch.utils.data.Subset(dts_images, pos_index)
    neg_dataset = torch.utils.data.Subset(dts_images, neg_index)

    quantifiers = ['CC', 'ACC', 'X', 'MAX', 'T50', 'MS', 'DyS', 'HDy', 'SORD', 
                   'MS2', 'EMQ', 'FM', 'FMM', 'SMM']

    test_sizes = [10, 20, 30, 40, 50, 100]
    alpha_values = [round(x, 2) for x in np.linspace(0, 1, 21)]
    print(alpha_values)
    iterations = 10

    arrayTPRFPR = Quantifier_Utils.TPRandFPR(validation_scores=val_scores)

    for size in test_sizes:
        for alpha in alpha_values:
            for iteration in range(iterations):
                pos_size = int(np.round(size * alpha, 2))
                pos_index = random.sample(range(len(pos_dataset)), pos_size)

                neg_size = size - pos_size
                neg_index = random.sample(range(len(neg_dataset)), neg_size)

                pos_subset = torch.utils.data.Subset(pos_dataset, pos_index)
                neg_subset = torch.utils.data.Subset(neg_dataset, neg_index)

                test_dataset = torch.utils.data.ConcatDataset([pos_subset, neg_subset])
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, num_workers=2, shuffle=True)

                pos_proportion = round(pos_size/size, 2)

                model.eval()
                with torch.no_grad():
                    probabilities = []
                    labels_list = []
                    for inputs, labels in test_dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        new_probabilities = torch.nn.functional.softmax(outputs, 1)

                        # Transforming to list and using only 4 decimals
                        new_probabilities = new_probabilities.tolist()
                        new_probabilities = [[round(prob, 4) for prob in row] for row in new_probabilities]

                        labels = labels.tolist()

                        labels_list.append(labels)
                        probabilities.append(new_probabilities)

                    # Transforming in only a list of probabilities (2 dimensions)
                    probabilities = [prob for prob_list in probabilities for prob in prob_list]
                    labels_list = [lab for lab_list in labels_list for lab in lab_list]

                    pos_probabilities = [prob[1] for prob in probabilities]

                    for quantifier in quantifiers:
                        predicted_proportion = run_quantifier(quantifier, probabilities, arrayTPRFPR, val_scores) # Added val_scores
                        pos_prediction = predicted_proportion[1]

                        accuracy, thr, accuracy2, thr2 = classifier_accuracy(pos_prediction, pos_probabilities, labels_list)

                        abs_error = round(abs(pos_prediction - pos_proportion), 2)

                        f_score = run_fscore(pos_probabilities, labels_list, thr)

                        instance = {'sample': iteration+1, 'test_size': size, 'alpha': alpha,
                                    'actual_prop': pos_proportion, 'pred_prop': pos_prediction,
                                    'abs_error': abs_error, 'acc': accuracy, 'f_score': f_score, 'thr': thr,
                                    'acc2': accuracy2, 'thr2': thr2, 'quantifier': quantifier, 'model': model_name, 'pt': boolean_pt}

                        instance = pd.DataFrame([instance])
                        result_table = pd.concat([result_table, instance], ignore_index=True)

                        print(result_table)

            # result_path definido no main!
            result_table.to_csv(result_path + '/ResultTable_' + model_name + '.csv', index=False)
            

    return result_table

def run_experiment(exp_name, model_name):
    dataset_images = transform_images('tests/' + exp_name)

    gs_df = pd.read_csv('gridSearch/' + exp_name + '/' + model_name + '/gridSearch.csv')

    gs_100 = gs_df[gs_df['epochs'] == 100]
    min_loss_config_index = gs_100['best_loss'].idxmin()
    min_loss_config = gs_100.loc[min_loss_config_index]

    best_lr = min_loss_config['lr']
    best_epochs = min_loss_config['epochs']
    best_pt = min_loss_config['pt']

    print(f'best lr: {best_lr}, best epochs: {best_epochs} best pt: {best_pt}')

    state_dict = torch.load('models/' + exp_name + '/' + model_name + '_' + exp_name + '.pth')
    model_cnn = setting_model(model_name, best_pt)

    model_cnn.load_state_dict(state_dict)

    validation_scores = pd.read_csv('scores/' + exp_name + '/' + model_name + 
                                    '/ValidationScores_' + str(best_lr) + '_' + str(best_epochs) + '_' + str(best_pt) + '.csv')

    result = experiment(model_cnn, dataset_images, validation_scores, model_name, best_pt) # Added model_name as a parameter!

    # result_path definido no main!
    result.to_csv(result_path + '/ResultTable_' + model_name + '.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='Experiment name')
    parser.add_argument('-m', '--model', default='resnet50', type=str, help='CNN model')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    exp_name = args.experiment
    model_name = args.model

    result_path = os.path.abspath('results/' + exp_name + '/')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    run_experiment(exp_name, model_name)
    print('Experiment ', exp_name, model_name, 'concluded!')
