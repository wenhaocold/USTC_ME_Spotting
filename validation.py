import os
import glob
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import yaml

from model import PEM
from post_process import calculate_proposal_with_score, nms
from tool import set_seed


def abfcm_nms_once(opt, epoch):
    def tmp_cvt(df):
        me_type = df.iloc[0].type
        df["type"] = "me" if me_type == 2 else "mae"
        return df
    abfcm_nms_csv = os.path.join(
        opt['project_root'], opt['output_dir_name'],
        'abfcm_nms', f'{opt["result_csv_name"]}.csv')
    dir_path, _ = os.path.split(abfcm_nms_csv)
    if os.path.exists(dir_path) is not True:
        os.makedirs(dir_path)
    abfcm_out_csv = os.path.join(
        opt['project_root'], opt['output_dir_name'],
        "abfcm_out",
        str(epoch).zfill(3) + ".csv")

    if os.path.exists(abfcm_nms_csv):
        os.remove(abfcm_nms_csv)

    if os.path.exists(abfcm_out_csv):
        df = pd.read_csv(abfcm_out_csv)
        # df["tp_fp"] = np.array(["FP"]*len(df))
        df = df.groupby(['video_name', "type_idx"]).apply(
            lambda x: nms(x, opt)).reset_index(drop=True)
        df = df.drop(["score"], axis=1)
        df = df.rename(columns={"video_name": "vid",
                                "start_frame": "pred_onset",
                                "end_frame": "pred_offset",
                                "type_idx": "type"})
        df = df.groupby("type").apply(tmp_cvt).reset_index(drop=True)
        if opt["dataset"].split('_')[1] == "samm":
            df["pred_onset"] = df["pred_onset"] * 8
            df["pred_offset"] = df["pred_offset"] * 8
        df.to_csv(abfcm_nms_csv, index=False)


def _get_model_output_full(model, epoch, device, opt, dataset):
    def _cal_proposal(array_softmax_score, array_apex_score, video_len,
                      video_name, type_idx):
        array_score_micro_start = array_softmax_score[:, 0, :].squeeze()
        array_score_micro_end = array_softmax_score[:, 1, :].squeeze()
        # array_score_micro_none = softmax_score[:,2,:].squeeze()

        proposal_block = calculate_proposal_with_score(
            array_score_micro_start, array_score_micro_end,
            array_apex_score, video_len, type_idx, opt)
        if proposal_block is not None:
            array_video_name = np.array(
                [video_name] * len(proposal_block)).reshape(-1, 1)
            array_type_idx = np.array(
                [type_idx] * len(proposal_block)).reshape(-1, 1)
            proposal_block = np.concatenate(
                (array_video_name, proposal_block, array_type_idx), axis=1)
        return proposal_block

    model.eval()

    predict_file = os.path.join(
        opt['project_root'], opt['output_dir_name'],
        'abfcm_out', str(epoch).zfill(3) + '.csv')
    dir_path, _ = os.path.split(predict_file)
    if os.path.exists(dir_path) is not True:
        os.makedirs(dir_path)
    feature_path_list = glob.glob(os.path.join(opt["feature_root"], "*.npy"))

    if os.path.exists(predict_file):
        os.remove(predict_file)

    tmp_array = []

    col_name = ["video_name", "start_frame", "end_frame", "start_socre",
                "end_score", "apex_score", "type_idx"]

    for feature_path in feature_path_list:
        feature = np.load(feature_path)
        video_name = os.path.split(feature_path)[-1].split('.')[0]
        feature = torch.from_numpy(feature).float()
        t, n, c = feature.shape
        feature = feature.reshape(1, t, n, c)

        feature[:, :, 0] = (feature[:, :, 0] - 0.002599) / 0.424943
        feature[:, :, 1] = (feature[:, :, 1] - 0.002586) / 0.421969

        # # casme
        # if dataset == "validation_casme":
        #     feature[:, :, 0] = (feature[:, :, 0] - 0.002195) / 0.406134
        #     feature[:, :, 1] = (feature[:, :, 1] - 0.002205) / 0.408799
        # # # samm
        # elif dataset == "validation_samm":
        #     feature[:, :, 0] = (feature[:, :, 0] - 0.003489) / 0.463610
        #     feature[:, :, 1] = (feature[:, :, 1] - 0.003423) / 0.449562
        # else:
        #     assert "feature normolization error"

        feature = feature.reshape(1, -1, 24).permute(0, 2, 1).contiguous()
        feature = feature.to(device)
        video_len = t
        output_probability = model(feature)

        output_micro_apex = output_probability[:, 6, :]
        output_macro_apex = output_probability[:, 7, :]
        output_micro_start_end = output_probability[:, 0: 0 + 3, :]
        output_macro_start_end = output_probability[:, 3: 3 + 3, :]

        # micro expression
        array_softmax_score_micro = torch.softmax(
            output_micro_start_end.cpu(), dim=1).numpy()
        array_score_micro_apex = torch.sigmoid(
            output_micro_apex.cpu()).squeeze().numpy()
        micro_proposal_block = _cal_proposal(
            array_softmax_score_micro, array_score_micro_apex, video_len,
            video_name, type_idx=2)
        if micro_proposal_block is not None:
            tmp_array.append(micro_proposal_block)

        # macro expression
        array_softmax_score_macro = torch.softmax(
            output_macro_start_end.cpu(), dim=1).numpy()
        array_score_macro_apex = torch.sigmoid(
            output_macro_apex.cpu()).squeeze().numpy()
        macro_proposal_block = _cal_proposal(
            array_softmax_score_macro, array_score_macro_apex, video_len,
            video_name, type_idx=1)
        if macro_proposal_block is not None:
            tmp_array.append(macro_proposal_block)

    if len(tmp_array) > 0:
        tmp_array = np.concatenate(tmp_array, axis=0)
        new_df = pd.DataFrame(tmp_array, columns=col_name)
        # TODO: the score should be a weighted sum
        new_df["score"] = new_df.start_socre * new_df.end_score * new_df.apex_score
        new_df = new_df.groupby('video_name').apply(
            lambda x: x.sort_values("score", ascending=False)).reset_index(drop=True)

        if len(new_df) > 0:
            new_df.to_csv(predict_file, index=False)


def validation(opt, dataset):
    device = opt['device'] if torch.cuda.is_available() else 'cpu'
    model = PEM(opt)
    model = model.to(device)

    best_epoch = opt["best_epoch"]

    weight_file = os.path.join(opt["model_save_root"], "abfcm_models",
                               "model_" + str(best_epoch).zfill(3) + ".pth")

    assert os.path.exists(weight_file), "weight file does not exist"
    print(f"weight_file = {weight_file}")

    checkpoint = torch.load(weight_file,
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        _get_model_output_full(model, best_epoch, device, opt, dataset)
    abfcm_nms_once(opt, best_epoch)


if __name__ == "__main__":
    set_seed(seed=42)

    parser = ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()
    assert args.dataset is not None
    dataset = args.dataset
    assert dataset == "validation_casme" or dataset == "validation_samm"

    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        opt = yaml_config[dataset]

    if opt["dataset"].split('_')[0] != "validation":
        print("dataset should be validation")
        exit(1)
    print(f'dataset: {opt["dataset"]}')
    validation(opt, dataset)
