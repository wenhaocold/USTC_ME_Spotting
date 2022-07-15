import numpy as np
import pandas as pd


def calculate_proposal_with_score(array_score_start, array_score_end,
                                  array_score_apex, video_len,
                                  type_idx, opt):
    ret = None
    if type_idx == 2:
        STEP = int(opt["RECEPTIVE_FILED"] // 2)  # int(opt["micro_average_len"]*3/2)
        EX_MIN = int(opt["micro_min"] // 2)
        apex_score_threshold = opt["micro_apex_score_threshold"]
    elif type_idx == 1:
        STEP = int(opt["macro_average_len"] // 2)  # int(opt["micro_average_len"]*3/2)
        EX_MIN = int(opt["macro_min"] // 2)
        apex_score_threshold = opt["macro_apex_score_threshold"]
    else:
        raise f"type_idx: {type_idx} is invalid"
    apex_indices = np.nonzero(array_score_apex > apex_score_threshold)[0]

    if apex_indices.size > 0:
        _tmp = np.arange(EX_MIN, STEP + 1, dtype=np.int64).reshape(1, -1)
        # _tmp = np.array([2, 3, 4, 5, 6, 7], dtype=np.int64).reshape(1, -1)
        start_indices = np.maximum(apex_indices.reshape(-1, 1) - _tmp, 0)
        start_indices_indices = np.argmax(array_score_start[start_indices],
                                          axis=-1)
        start_indices = start_indices[np.arange(start_indices.shape[0]),
                                      start_indices_indices]
        end_indices = np.minimum(apex_indices.reshape(-1, 1) + _tmp,
                                 video_len - 1)
        end_indices_indices = np.argmax(array_score_end[end_indices], axis=-1)
        end_indices = end_indices[np.arange(end_indices.shape[0]),
                                  end_indices_indices]

        start_indices_list = []
        end_indices_list = []
        apex_indices_list = []
        for k in range(start_indices.size):
            start_index = start_indices[k].item()
            end_index = end_indices[k].item()
            apex_index = apex_indices[k].item()
            if (array_score_start[start_index] > array_score_end[start_index]) \
            and(array_score_start[end_index]   < array_score_end[end_index]):# \
            # and(tmp_score_start[start_index] > tmp_score_none[start_index]) \
            # and(tmp_score_end[end_index]     > tmp_score_none[end_index]):
                start_indices_list.append(start_index)
                end_indices_list.append(end_index)
                apex_indices_list.append(apex_index)

        if len(start_indices_list) > 0:
            start_indices = np.array(start_indices_list)
            end_indices = np.array(end_indices_list)
            apex_indices = np.array(apex_indices_list)
            ret = (np.array([
                            # [video_name]*len(start_indices),
                            start_indices, end_indices,
                            array_score_start[start_indices],
                            array_score_end[end_indices],
                            array_score_apex[apex_indices]],
                            dtype=object).T)
    return ret


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = (anchors_max - anchors_min) + (box_max - box_min) - inter_len
    iou = np.divide(inter_len, union_len)
    return iou


def nms(df, opt):
    tstart = list(df.start_frame.values)
    tend = list(df.end_frame.values)
    tscore = list(df.score.values)
    video_name = df.video_name.values[0]
    type_idx = df.type_idx.values[0]

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 0 and len(rscore) < opt['nms_top_K']:
        max_index = np.argmax(tscore)
        if (tscore[max_index]) == 0:
            break
        iou_list = iou_with_anchors(
            tstart[max_index], tend[max_index],
            np.array(tstart), np.array(tend))
        # iou_exp_list = np.exp(-np.square(iou_list)/0.75)
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = iou_list[idx]
                if tmp_iou > 0.5:
                    tscore[idx] = 0  # tscore[idx]*iou_exp_list[idx]

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])

        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['video_name'] = [video_name] * len(rstart)
    newDf['start_frame'] = rstart
    newDf['end_frame'] = rend
    newDf['score'] = rscore
    newDf["type_idx"] = [type_idx] * len(rstart)
    return newDf


def iou_for_find(df, opt):
    video_name = df.video_name.values[0]
    type_idx = df.type_idx.values[0]

    anno_df = pd.read_csv(opt['anno_csv'])
    tmp_anno_df = anno_df[anno_df['video_name'] == video_name]
    tmp_anno_df = tmp_anno_df[tmp_anno_df['type_idx'] == type_idx]

    gt_start_indices = tmp_anno_df.start_frame.values // opt["RATIO_SCALE"]
    gt_end_indices = tmp_anno_df.end_frame.values // opt["RATIO_SCALE"]
    predict_start_indices = df.start_frame.values
    predict_end_indices = df.end_frame.values

    tiou = np.array([0.] * len(df))
    idx_list = []

    for j in range(len(gt_start_indices)):
        gt_start = gt_start_indices[j]
        gt_end = gt_end_indices[j]
        ious = iou_with_anchors(gt_start, gt_end, predict_start_indices, predict_end_indices)
        max_iou = max(ious)
        if max_iou > 0.5:
            tmp_idx = np.argmax(ious)
            idx_list.append(tmp_idx)
            tiou[tmp_idx] = max_iou

    find = np.array(["False"] * len(df))
    find[idx_list] = "True"
    df["find"] = find
    df["iou"] = tiou
    return df


def iou_for_tp(df, opt):
    video_name = df.video_name.values[0]
    type_idx = df.type_idx.values[0]

    anno_df = pd.read_csv(opt['anno_csv'])
    tmp_anno_df = anno_df[anno_df['video_name'] == video_name]
    tmp_anno_df = tmp_anno_df[tmp_anno_df['type_idx'] == type_idx]

    if len(tmp_anno_df) == 0:
        tp = np.array(["False"] * len(df))
        df["tp"] = tp
        df["iou"] = 0
        return df

    gt_start_indices = tmp_anno_df.start_frame.values // opt["RATIO_SCALE"]
    gt_end_indices = tmp_anno_df.end_frame.values // opt["RATIO_SCALE"]
    predict_start_indices = df.start_frame.values
    predict_end_indices = df.end_frame.values

    tiou = np.array([0.] * len(df))
    idx_list = []

    for j in range(len(predict_start_indices)):
        predict_start = predict_start_indices[j]
        predict_end = predict_end_indices[j]
        ious = iou_with_anchors(predict_start, predict_end, gt_start_indices, gt_end_indices)
        max_iou = max(ious)
        if max_iou > 0.5:
            tiou[j] = max_iou
            idx_list.append(j)

    tp_fp = np.array(["False"] * len(df))
    tp_fp[idx_list] = "True"
    df["tp"] = tp_fp
    df["iou"] = tiou
    return df
