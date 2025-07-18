import copy
import math
import random

import numpy
import torch
import torchvision


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def smooth(y, f=0.1):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(px, py, ap, names, save_dir):
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = numpy.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    pyplot.close(fig)


def plot_curve(px, py, names, save_dir, x_label="Confidence", y_label="Metric"):
    from matplotlib import pyplot

    figure, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), f=0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.3f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{y_label}-Confidence Curve")
    figure.savefig(save_dir, dpi=250)
    pyplot.close(figure)


def compute_ap(tp, conf, output, target, plot=False, names=(), eps=1E-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        output:  Predicted object classes (nparray).
        target:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, output = tp[i], conf[i], output[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(start=0, stop=1, num=1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = output == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(start=0, stop=1, num=101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
            if plot and j == 0:
                py.append(numpy.interp(px, m_rec, m_pre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    if plot:
        names = dict(enumerate(names))  # to dict
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        plot_pr_curve(px, py, ap, names, save_dir="./weights/PR_curve.png")
        plot_curve(px, f1, names, save_dir="./weights/F1_curve.png", y_label="F1")
        plot_curve(px, p, names, save_dir="./weights/P_curve.png", y_label="Precision")
        plot_curve(px, r, names, save_dir="./weights/R_curve.png", y_label="Recall")
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    f1 = f1.mean()
    m_pre = p.mean()
    m_rec = r.mean()
    map50 = ap50.mean()
    mean_ap = ap.mean()
    return tp, fp, m_pre, m_rec, f1, map50, mean_ap


def disable_grad(filename):
    x = torch.load(filename, map_location="cpu")
    for p in x['model'].parameters():
        p.requires_grad_ = False
    torch.save(x, f=filename)


def clip_gradients(model, max_norm=10):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def load_weight(model, ckpt):
    dst = model.state_dict()
    src = torch.load(ckpt)['model'].float().cpu()

    ckpt = {}
    for k, v in src.state_dict().items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v

    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def set_params(model, decay):
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if not p.requires_grad:
                continue
            if n == "bias":  # bias (no decay)
                p1.append(p)
            elif n == "weight" and isinstance(m, norm):  # norm-weight (no decay)
                p1.append(p)
            else:
                p2.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


def plot_lr(args, optimizer, scheduler, num_steps):
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        for i in range(num_steps):
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)
            y.append(optimizer.param_groups[0]['lr'])
    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('step')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs * num_steps)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


class CosineLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 1500))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps))

        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class LinearLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 1500))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = numpy.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class QualityFocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 2.0
        self.epsilon = torch.finfo(torch.float32).eps
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, target, weight=None, avg_factor=1):
        # label denotes the category id, score denotes the quality score
        label, score = target
        # negatives are supervised by 0 quality score
        output_sigmoid = output.sigmoid()
        scale_factor = output_sigmoid
        zero_label = scale_factor.new_zeros(output.shape)
        loss = self.bce_loss(output, zero_label) * scale_factor.pow(self.beta)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = output.size(1)
        pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
        pos_label = label[pos].long()
        # positives are supervised by bbox quality (IoU) score
        scale_factor = score[pos] - output_sigmoid[pos, pos_label]
        loss[pos, pos_label] = self.bce_loss(output[pos, pos_label],
                                             score[pos]) * scale_factor.abs().pow(self.beta)

        loss = loss.sum(dim=1, keepdim=False)
        if weight is not None:
            loss = loss * weight

        return loss.sum() / (avg_factor + self.eps)


class BoxLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, loss_weight=5.0, smooth_point=0.1):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self, output, target):
        px1, py1, px2, py2 = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # extent top left
        ex1 = torch.min(px1, tx1)
        ey1 = torch.min(py1, ty1)

        # intersection coordinates
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)

        # extra
        x_min = torch.min(ix1, ix2)
        y_min = torch.min(iy1, iy2)
        x_max = torch.max(ix1, ix2)
        y_max = torch.max(iy1, iy2)

        # Intersection
        intersection = (ix2 - ex1) * (iy2 - ey1) + (x_min - ex1) * (y_min - ey1) - (
                ix1 - ex1) * (y_max - ey1) - (x_max - ex1) * (iy1 - ey1)
        # Union
        union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - intersection + self.eps
        # IoU
        iou = 1 - (intersection / union)

        # Smooth-EIoU
        smooth_sign = (iou < self.smooth_point).detach().float()
        loss = 0.5 * smooth_sign * (iou ** 2) / self.smooth_point + (1 - smooth_sign) * (iou - 0.5 * self.smooth_point)
        return self.loss_weight * (loss.sum())


class SimOTAAssigner:
    EPS = 1E-7
    INF = 100000.0

    def __init__(self, candidate_top_k=13, center_radius=2.5, iou_weight=3.0, cls_weight=1.0):
        self.candidate_top_k = candidate_top_k
        self.center_radius = center_radius
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def __call__(self, x_cls, x_box, x_obj, y_box, y_cls, anchors):
        num_gt = y_box.size(0)
        num_bboxes = x_box.size(0)
        x_cls_sigmoid = x_cls.sigmoid() * x_obj.unsqueeze(1).sigmoid()

        # assign 0 by default
        assigned_gt_indices = x_box.new_full((num_bboxes,), 0, dtype=torch.long)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(anchors, y_box)

        valid_boxes = x_box[valid_mask]
        valid_scores = x_cls_sigmoid[valid_mask]

        num_valid = valid_boxes.size(0)
        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = x_box.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_indices[:] = 0
            if y_cls is None:
                assigned_labels = None
            else:
                assigned_labels = x_box.new_full((num_bboxes,), -1, dtype=torch.long)
            return assigned_gt_indices, max_overlaps, assigned_labels

        pairwise_iou = self.compute_iou(valid_boxes, y_box)
        iou_cost = -torch.log(pairwise_iou + self.EPS)

        gt_onehot_label = (torch.nn.functional.one_hot(y_cls.to(torch.int64),
                                                       x_cls_sigmoid.shape[-1]
                                                       ).float().unsqueeze(0).repeat(num_valid, 1, 1))

        dtype = valid_scores.dtype
        valid_scores = valid_scores.unsqueeze(1).repeat(1, num_gt, 1)
        valid_scores = valid_scores.to(dtype=torch.float32).sqrt_()
        with torch.amp.autocast(device_type='cuda', enabled=False):
            cls_cost = (torch.nn.functional.binary_cross_entropy(valid_scores,
                                                                 gt_onehot_label,
                                                                 reduction='none').sum(-1).to(dtype=dtype))

        cost_matrix = (cls_cost * self.cls_weight
                       + iou_cost * self.iou_weight +
                       (~is_in_boxes_and_center) * self.INF)
        matched_iou, matched_indices = self.dynamic_k_matching(cost_matrix, pairwise_iou, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_indices[valid_mask] = matched_indices + 1
        assigned_labels = assigned_gt_indices.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = y_cls[matched_indices].long()
        max_overlaps = assigned_gt_indices.new_full((num_bboxes,), -self.INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_iou
        return assigned_gt_indices, max_overlaps, assigned_labels

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_iou, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate top_k iou for dynamic-k calculation
        candidate_top_k = min(self.candidate_top_k, pairwise_iou.size(0))
        top_k_iou, _ = torch.topk(pairwise_iou, candidate_top_k, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(top_k_iou.sum(0).int(), min=1)
        for i in range(num_gt):
            _, pos_idx = torch.topk(cost[:, i], k=dynamic_ks[i], largest=False)
            matching_matrix[:, i][pos_idx] = 1

        del top_k_iou, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_indices = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_iou = (matching_matrix * pairwise_iou).sum(1)[fg_mask_inboxes]
        return matched_iou, matched_indices

    def compute_iou(self, boxes1, boxes2, eps=1e-6):
        # Either the boxes are empty or the length of boxes' last dimension is 4
        assert (boxes1.size(-1) == 4 or boxes1.size(0) == 0)
        assert (boxes2.size(-1) == 4 or boxes2.size(0) == 0)

        # Batch dim must be the same
        # Batch dim: (B1, B2, ... Bn)
        assert boxes1.shape[:-2] == boxes2.shape[:-2]
        batch_shape = boxes1.shape[:-2]

        rows = boxes1.size(-2)
        cols = boxes2.size(-2)

        if rows * cols == 0:
            return boxes1.new(batch_shape + (rows, cols))

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = self.clamp(rb - lt, a=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap
        union = torch.max(union, union.new_tensor([eps]))

        return overlap / union

    @staticmethod
    def clamp(x, a=None, b=None):
        if not x.is_cuda and x.dtype == torch.float16:
            return x.float().clamp(a, b).half()
        return x.clamp(a, b)


class DSLAssigner:
    INF = 100000000

    def __init__(self, top_k=13, iou_factor=3.0):
        self.top_k = top_k
        self.iou_factor = iou_factor

    def __call__(self, x_cls, x_box, x_obj, y_box, y_cls, anchors):
        num_gt = y_box.size(0)
        num_bboxes = x_box.size(0)
        x_cls_sigmoid = x_cls.sigmoid() * x_obj.unsqueeze(1).sigmoid()
        # assign 0 by default
        assigned_gt_indices = x_box.new_full((num_bboxes,), 0, dtype=torch.long)

        prior_center = anchors[:, :2]
        lt_ = prior_center[:, None] - y_box[:, :2]
        rb_ = y_box[:, 2:] - prior_center[:, None]

        deltas = torch.cat(tensors=[lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = x_box[valid_mask]
        valid_scores = x_cls_sigmoid[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = x_box.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_indices[:] = 0
            if y_cls is None:
                assigned_labels = None
            else:
                assigned_labels = x_box.new_full((num_bboxes,), -1, dtype=torch.long)
            return assigned_gt_indices, max_overlaps, assigned_labels

        pairwise_iou = self.compute_iou(valid_decoded_bbox, y_box)
        iou_cost = -torch.log(pairwise_iou + 1e-7)

        gt_onehot_label = (torch.nn.functional.one_hot(y_cls.to(torch.int64),
                                                       x_cls_sigmoid.shape[-1])
                           .float()
                           .unsqueeze(0)
                           .repeat(num_valid, 1, 1))
        valid_scores = valid_scores.unsqueeze(1).repeat(1, num_gt, 1)
        soft_label = gt_onehot_label * pairwise_iou[..., None]
        scale_factor = soft_label - valid_scores.sigmoid()

        cls_cost = torch.nn.functional.binary_cross_entropy_with_logits(valid_scores,
                                                                        soft_label,
                                                                        reduction="none"
                                                                        ) * scale_factor.abs().pow(2.0)

        cls_cost = cls_cost.sum(dim=-1)

        cost_matrix = cls_cost + iou_cost * self.iou_factor
        matched_iou, matched_indices = self.dynamic_k_matching(cost_matrix,
                                                               pairwise_iou,
                                                               num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_indices[valid_mask] = matched_indices + 1
        assigned_labels = assigned_gt_indices.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = y_cls[matched_indices].long()
        max_overlaps = assigned_gt_indices.new_full((num_bboxes,), -self.INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_iou

        return assigned_gt_indices, max_overlaps, assigned_labels

    def dynamic_k_matching(self, cost, pairwise_iou, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost)
        # select candidate top_k iou for dynamic-k calculation
        candidate_top_k = min(self.top_k, pairwise_iou.size(0))
        top_k_iou, _ = torch.topk(pairwise_iou, candidate_top_k, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(top_k_iou.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del top_k_iou, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_indices = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_iou = (matching_matrix * pairwise_iou).sum(1)[fg_mask_inboxes]
        return matched_iou, matched_indices

    @staticmethod
    def compute_iou(boxes1, boxes2, eps=1e-6):
        assert boxes1.size(-1) == 4 or boxes1.size(0) == 0
        assert boxes2.size(-1) == 4 or boxes2.size(0) == 0

        # Batch dim must be the same
        # Batch dim: (B1, B2, ... Bn)
        assert boxes1.shape[:-2] == boxes2.shape[:-2]
        batch_shape = boxes1.shape[:-2]

        rows = boxes1.size(-2)
        cols = boxes2.size(-2)

        if rows * cols == 0:
            return boxes1.new(batch_shape + (rows, cols))

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        return overlap / union


class ComputeLoss:
    def __init__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        m = model.head

        self.nc = m.nc

        self.strides = m.strides

        self.loss_box = BoxLoss()
        self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.assigner = SimOTAAssigner()

    def __call__(self, outputs, targets):
        x_cls, x_box, x_obj = outputs
        assert len(x_cls) == len(x_box) == len(x_obj)

        n = outputs[0][0].shape[0]
        sizes = [i.shape[2:] for i in x_cls]
        anchors = self.__make_anchors(sizes, self.strides, x_cls[0].device, x_cls[0].dtype)
        anchors = torch.cat(anchors).unsqueeze(0).repeat(n, 1, 1)

        x_cls = [i.permute(0, 2, 3, 1).reshape(n, -1, self.nc) for i in x_cls]
        x_box = [i.permute(0, 2, 3, 1).reshape(n, -1, 4) for i in x_box]
        x_obj = [i.permute(0, 2, 3, 1).reshape(n, -1) for i in x_obj]

        x_cls = torch.cat(x_cls, dim=1)
        x_box = torch.cat(x_box, dim=1)
        x_obj = torch.cat(x_obj, dim=1)
        x_box = self.__box_decode(anchors, x_box)

        pos_masks = []
        cls_targets = []
        obj_targets = []
        box_targets = []
        num_fg_images = []

        x_cls_detached = x_cls.detach()
        x_obj_detached = x_obj.detach()
        x_box_detached = x_box.detach()

        for i in range(n):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]
            target = self.__get_target(x_cls_detached[i],
                                       x_obj_detached[i],
                                       x_box_detached[i],
                                       box, cls, anchors[i])
            pos_masks.append(target[0])
            cls_targets.append(target[1])
            obj_targets.append(target[2])
            box_targets.append(target[3])
            num_fg_images.append(target[4])

        pos_masks = torch.cat(pos_masks, dim=0)
        cls_targets = torch.cat(cls_targets, dim=0)
        obj_targets = torch.cat(obj_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)

        num_pos = torch.tensor(sum(num_fg_images),
                               dtype=torch.float,
                               device=x_cls.device)
        num_total_samples = max(self.__reduce_mean(num_pos), 1.0)

        loss_box = self.loss_box(x_box.view(-1, 4)[pos_masks],
                                 box_targets) / num_total_samples
        loss_obj = self.loss_bce(x_obj.view(-1, 1),
                                 obj_targets.float()).sum() / num_total_samples
        loss_cls = self.loss_bce(x_cls.view(-1, self.nc)[pos_masks],
                                 cls_targets.float()).sum() / num_total_samples

        return loss_cls, loss_box, loss_obj

    @staticmethod
    def __box_decode(priors, box):
        xys = (box[..., :2] * priors[..., 2:]) + priors[..., :2]
        whs = box[..., 2:].exp() * priors[..., 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        return torch.stack(tensors=[tl_x, tl_y, br_x, br_y], dim=-1)

    @torch.no_grad()
    def __get_target(self, x_cls, x_obj, x_box, y_box, y_cls, priors):
        num_priors = priors.size(0)
        num_gts = y_cls.size(0)
        y_cls = y_cls.to(x_box.device)
        y_box = y_box.to(x_box.dtype)
        y_box = y_box.to(x_box.device)
        # No target
        if num_gts == 0:
            cls_target = x_cls.new_zeros((0, self.nc))
            box_target = x_cls.new_zeros((0, 4))
            obj_target = x_cls.new_zeros((num_priors, 1))
            foreground_mask = x_cls.new_zeros(num_priors).bool()
            return foreground_mask, cls_target, obj_target, box_target, 0

        # Uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        gt_indices, max_overlaps, labels = self.assigner(x_cls, x_box, x_obj, y_box, y_cls, offset_priors)

        indices = torch.nonzero(gt_indices > 0, as_tuple=False).squeeze(-1).unique()

        pos_assigned_gt_indices = gt_indices[indices] - 1

        if y_box.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_indices.numel() == 0
            bbox_target = torch.empty_like(y_box).view(-1, 4)
        else:
            if len(y_box.shape) < 2:
                y_box = y_box.view(-1, 4)

            bbox_target = y_box[pos_assigned_gt_indices.long(), :]

        if labels is not None:
            pos_gt_labels = labels[indices]
        else:
            pos_gt_labels = None

        num_pos_per_img = indices.size(0)

        cls_target = torch.nn.functional.one_hot(pos_gt_labels,
                                                 self.nc) * max_overlaps[indices].unsqueeze(-1)

        obj_target = torch.zeros_like(x_obj).unsqueeze(-1)
        obj_target[indices] = 1

        foreground_mask = torch.zeros_like(x_obj).to(torch.bool)
        foreground_mask[indices] = 1
        return foreground_mask, cls_target, obj_target, bbox_target, num_pos_per_img

    @staticmethod
    def __reduce_mean(x):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return x
        x = x.clone()
        torch.distributed.all_reduce(x.div_(torch.distributed.get_world_size()),
                                     op=torch.distributed.ReduceOp.SUM)
        return x

    @staticmethod
    def __make_anchors(sizes, strides, device, dtype, offset=0.0):
        anchors = []
        assert len(sizes) == len(strides)
        for stride, size in zip(strides, sizes):
            # keep size as Tensor instead of int, so that we can convert to ONNX correctly
            shift_x = ((torch.arange(0, size[1]) + offset) * stride).to(dtype)
            shift_y = ((torch.arange(0, size[0]) + offset) * stride).to(dtype)

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
            stride_w = shift_x.new_full((shift_x.shape[0],), stride).to(dtype)
            stride_h = shift_x.new_full((shift_y.shape[0],), stride).to(dtype)
            anchors.append(torch.stack(tensors=[shift_x, shift_y, stride_w, stride_h], dim=-1).to(device))
        return anchors


class NMS:
    def __init__(self, conf_threshold=0.001, iou_threshold=0.7):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, outputs):
        x_cls, x_box, x_obj = outputs
        assert len(x_cls) == len(x_box) == len(x_obj)

        outputs = []
        for i in range(x_cls.shape[0]):
            cls = x_cls[i]
            obj = x_obj[i]
            box = x_box[i]
            outputs.append(self.__nms(cls, box, obj))

        return outputs

    def __nms(self, cls, box, obj):
        scores, indices = torch.max(cls, 1)
        valid_mask = obj * scores >= self.conf_threshold

        boxes = box[valid_mask]
        scores = scores[valid_mask] * obj[valid_mask]
        indices = indices[valid_mask]

        if indices.numel() == 0:
            return torch.cat(tensors=(boxes,
                                      scores[:, None],
                                      indices[:, None]), dim=-1)
        else:
            keep = torchvision.ops.batched_nms(boxes,
                                               scores,
                                               indices,
                                               self.iou_threshold)
            return torch.cat(tensors=(boxes[keep],
                                      scores[keep][:, None],
                                      indices[keep][:, None]), dim=-1)
