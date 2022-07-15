import os
import time
from datetime import datetime
import multiprocessing
from argparse import ArgumentParser

import yaml
import torch

from tool import set_seed
from abfcm import abfcm_train_group, abfcm_output_group, abfcm_nms, abfcm_iou_process, \
    abfcm_final_result_per_subject, abfcm_final_result_best, abfcm_train_and_eval


def bi_loss(output, label):
    weight = torch.empty_like(output)
    c_0 = 0.05  # (label > 0).sum / torch.numel(label)
    c_1 = 1 - c_0
    weight[label > 0] = c_1
    weight[label == 0] = c_0
    loss = torch.nn.functional.binary_cross_entropy(output, label, weight)
    return loss


def create_folder(opt):
    # create folder
    output_path = os.path.join(opt['project_root'], opt['output_dir_name'])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for subject in subject_list:
        out_subject_path = os.path.join(output_path, subject)
        if not os.path.exists(out_subject_path):
            os.mkdir(out_subject_path)
        subject_abfcm_out = os.path.join(out_subject_path, 'abfcm_out')
        if not os.path.exists(subject_abfcm_out):
            os.mkdir(subject_abfcm_out)
        subject_abfcm_nms_path = os.path.join(out_subject_path, 'abfcm_nms')
        if not os.path.exists(subject_abfcm_nms_path):
            os.mkdir(subject_abfcm_nms_path)
        subject_abfcm_final_result_path = os.path.join(
            out_subject_path, 'sub_abfcm_final_result')
        if not os.path.exists(subject_abfcm_final_result_path):
            os.mkdir(subject_abfcm_final_result_path)


def abfcm_train_mul_process(subject_group, opt):
    print("abfcm abfcm_train_mul_process ------ start: ")
    print("abfcm_training_lr: ", opt["abfcm_training_lr"])
    print("abfcm_weight_decay: ", opt["abfcm_weight_decay"])
    print("abfcm_lr_scheduler: ", opt["abfcm_lr_scheduler"])
    print("abfcm_apex_gamma: ", opt["abfcm_apex_gamma"])
    print("abfcm_apex_alpha: ", opt["abfcm_apex_alpha"])
    print("abfcm_action_gamma: ", opt["abfcm_action_gamma"])
    print("abfcm_action_alpha: ", opt["abfcm_action_alpha"])

    process = []
    start_time = datetime.now()
    for subject_list in subject_group:
        p = multiprocessing.Process(target=abfcm_train_group,
                                    args=(opt, subject_list))
        p.start()
        process.append(p)
        time.sleep(1)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("abfcm abfcm_train_mul_process ------ sucessed: ")
    print("time: ", delta_time)


def abfcm_output_mul_process(subject_group, opt):
    print("abfcm_output_mul_process ------ start: ")
    print("micro_apex_score_threshold: ", opt["micro_apex_score_threshold"])
    print("macro_apex_score_threshold: ", opt["macro_apex_score_threshold"])
    process = []
    start_time = datetime.now()
    for subject_list in subject_group:
        p = multiprocessing.Process(target=abfcm_output_group,
                                    args=(opt, subject_list))
        p.start()
        p.join()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("abfcm_output_mul_process ------ sucessed: ")
    print("time: ", delta_time)


def abfcm_nms_mul_process(subject_list, opt):
    print("abfcm_nms ------ start: ")
    print("nms_top_K: ", opt["nms_top_K"])
    process = []
    start_time = datetime.now()
    for subject in subject_list:
        p = multiprocessing.Process(target=abfcm_nms, args=(opt, subject))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("abfcm_nms ------ sucessed: ")
    print("time: ", delta_time)


def abfcm_iou_mul_process(subject_list, opt):
    print("abfcm_iou_process ------ start: ")
    process = []
    start_time = datetime.now()
    for subject in subject_list:
        p = multiprocessing.Process(target=abfcm_iou_process,
                                    args=(opt, subject))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    delta_time = datetime.now() - start_time
    print("abfcm_iou_process ------ sucessed: ")
    print("time: ", delta_time)


if __name__ == "__main__":
    set_seed(seed=42)

    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--mode")
    args = parser.parse_args()

    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    subject_list = opt['subject_list']

    if args.mode is not None:
        opt["mode"] = args.mode

    create_folder(opt)

    print(f"===================== Dataset is {dataset} =====================")

    if dataset != "cross":
        tmp_work_numbers = 5
        subject_group = []
        if len(subject_list) % tmp_work_numbers == 0:
            len_per_group = int(len(subject_list) // tmp_work_numbers)
            for i in range(tmp_work_numbers):
                subject_group.append(subject_list[
                                     i * len_per_group:
                                     (i + 1) * len_per_group])
        else:
            len_per_group = int(len(subject_list) // tmp_work_numbers) + 1
            last_len = len(subject_list) - len_per_group * (tmp_work_numbers - 1)
            for i in range(tmp_work_numbers - 1):
                subject_group.append(subject_list[
                                     i * len_per_group:
                                     (i + 1) * len_per_group])
            subject_group.append(subject_list[-last_len:])

        if opt["mode"] == "abfcm_train_mul_process":
            abfcm_train_mul_process(subject_group, opt)
        elif opt["mode"] == "abfcm_output_mul_process":
            abfcm_output_mul_process(subject_group, opt)
        elif opt["mode"] == "abfcm_nms_mul_process":
            abfcm_nms_mul_process(subject_list, opt)
        elif opt["mode"] == "abfcm_iou_mul_process":
            abfcm_iou_mul_process(subject_list, opt)
        elif opt["mode"] == "abfcm_final_result":
            print("abfcm_final_result ------ start: ")
            # abfcm_final_result(opt, subject_list)
            # smic doesn't have macro label
            if dataset != "smic":
                abfcm_final_result_per_subject(opt, subject_list, type_idx=1)
            abfcm_final_result_per_subject(opt, subject_list, type_idx=2)
            abfcm_final_result_per_subject(opt, subject_list, type_idx=0)
            abfcm_final_result_best(opt, subject_list, type_idx=0)
            # abfcm_final_result_best(opt, subject_list, type_idx=1)
            # abfcm_final_result_best(opt, subject_list, type_idx=2)
            print("abfcm_final_result ------ successed")
    else:
        abfcm_train_and_eval(opt)
