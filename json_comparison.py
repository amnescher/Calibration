import zipfile
import argparse
import numpy as np
import json
import glob 
import os
from evaluate_extremities import mirror_labels, evaluate_detection_prediction, scale_points


def evaluate(gt_path, prediction_path, width=960, height=540):

    
    predic_files = glob.glob(prediction_path+"/*.json")
    gt_files = glob.glob(gt_path+"/*.json")

    predic_list = []
    gt_list =[]

    for file in predic_files:
        predic_list.append(file.split("/")[-1])
    
    for file in gt_files:
        gt_list.append(file.split("/")[-1])


    accuracies = []
    precisions = []
    recalls = []
    dict_errors = {}
    per_class_confusion_dict = {}
    total_frames = 0
    missed = 0

    for gt_json in gt_list:
        if gt_json not in predic_list:
            accuracies.append(0.)
            precisions.append(0.)
            recalls.append(0.)
            continue

        prediction_Json_path = os.path.join(prediction_path,gt_json)
        with open(prediction_Json_path, 'r') as j:
            prediction = json.loads(j.read())
        
        
        gt_Json_path = os.path.join(gt_path,gt_json)
        with open(gt_Json_path,'r') as j:
            gt = json.loads(j.read())

        predictions = scale_points(prediction, width, height)
        line_annotations = scale_points(gt, width, height)

        img_prediction = predictions
        img_groundtruth = line_annotations
        if img_groundtruth == None or img_prediction==None:
            continue
        confusion1, per_class_conf1, reproj_errors1 = evaluate_detection_prediction(img_prediction,
                                                                                    img_groundtruth,
                                                                                    10)
        confusion2, per_class_conf2, reproj_errors2 = evaluate_detection_prediction(img_prediction,
                                                                                    mirror_labels(
                                                                                        img_groundtruth),
                                                                                    10)

        accuracy1, accuracy2 = 0., 0.
        if confusion1.sum() > 0:
            accuracy1 = confusion1[0, 0] / confusion1.sum()

        if confusion2.sum() > 0:
            accuracy2 = confusion2[0, 0] / confusion2.sum()

        if accuracy1 > accuracy2:
            accuracy = accuracy1
            confusion = confusion1
            per_class_conf = per_class_conf1
            reproj_errors = reproj_errors1
        else:
            accuracy = accuracy2
            confusion = confusion2
            per_class_conf = per_class_conf2
            reproj_errors = reproj_errors2

        accuracies.append(accuracy)
        if confusion[0, :].sum() > 0:
            precision = confusion[0, 0] / (confusion[0, :].sum())
            precisions.append(precision)
        if (confusion[0, 0] + confusion[1, 0]) > 0:
            recall = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])
            recalls.append(recall)

        for line_class, errors in reproj_errors.items():
            if line_class in dict_errors.keys():
                dict_errors[line_class].extend(errors)
            else:
                dict_errors[line_class] = errors

        for line_class, confusion_mat in per_class_conf.items():
            if line_class in per_class_confusion_dict.keys():
                per_class_confusion_dict[line_class] += confusion_mat
            else:
                per_class_confusion_dict[line_class] = confusion_mat
    results = {}
    results["meanRecall"] = np.mean(recalls)
    results["meanPrecision"] = np.mean(precisions)
    results["meanAccuracies"] = np.mean(accuracies)

    for line_class, confusion_mat in per_class_confusion_dict.items():
        class_accuracy = confusion_mat[0, 0] / confusion_mat.sum()
        class_recall = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
        class_precision = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
        results[f"{line_class}Precision"] = class_precision
        results[f"{line_class}Recall"] = class_recall
        results[f"{line_class}Accuracy"] = class_accuracy
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation camera calibration task')

    parser.add_argument('-s', '--soccernet', default="/home/ribble/Documents/Amin_dirc/sn-calibration/RTDataSet2/valid/Json", type=str,
                        help='Path to the zip groundtruth folder')
    parser.add_argument('-p', '--prediction', default="/home/ribble/Documents/Amin_dirc/sn-calibration/debug/results_Json",
                        required=False, type=str,
                        help="Path to the  zip prediction folder")

    args = parser.parse_args()

    accuracy = evaluate(args.soccernet, args.prediction,width=512, height=512)

    for name, val in accuracy.items():
        print(name, val)