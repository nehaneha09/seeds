import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.vit.utils.ops import HungarianMatcher
from ultralytics.yolo.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.yolo.utils.metrics import bbox_iou


class DETRLoss(nn.Module):

    def __init__(self,
                 nc=80,
                 matcher=HungarianMatcher(cost_gain={
                     'class': 2,
                     'bbox': 5,
                     'giou': 2}),
                 loss_gain=None,
                 aux_loss=True,
                 use_focal_loss=True,
                 use_vfl=False,
                 use_uni_match=False,
                 uni_match_ind=0):
        """
        Args:
            nc (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_gain (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
            use_vfl (bool): Use VarifocalLoss or not.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1}
        self.nc = nc
        self.matcher = matcher
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_focal_loss else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

        if not use_focal_loss:
            self.loss_gain['class'] = torch.full([nc + 1], loss_gain['class'])
            self.loss_gain['class'][-1] = loss_gain['no_object']

    def _get_loss_class(self, scores, gt_class, match_indices, num_gts, postfix='', iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f'loss_class{postfix}'
        bs, nq = scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=scores.device, dtype=gt_class.dtype)
        # NOTE: num_gt could be different from num_gts, because of the denoising part.
        idx, gt_idx = self._get_index(match_indices)
        # targets[idx] = torch.cat([t[dst].squeeze(-1) for t, (_, dst) in zip(gt_class, match_indices)])
        targets[idx] = gt_class[gt_idx]
        if self.fl:
            # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
            one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
            one_hot.scatter_(2, targets.unsqueeze(-1), 1)
            one_hot = one_hot[..., :-1]

            if iou_score is not None and self.vfl:
                gt_scores = torch.zeros([bs, nq], device=scores.device)
                gt_scores[idx] = iou_score
                gt_scores = gt_scores.view(bs, nq, 1) * one_hot
                loss_cls = self.vfl(scores, gt_scores, one_hot, num_gts / nq)
                # loss_ = nn.BCEWithLogitsLoss(reduction='none')(logits, target_score).mean(1).sum()  # YOLO CLS loss
            else:
                loss_cls = self.fl(scores, one_hot.float(), num_gts / nq)
        else:
            loss_cls = F.cross_entropy(scores, targets)

        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, num_gts, postfix=''):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f'loss_bbox{postfix}'
        name_giou = f'loss_giou{postfix}'

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0., device=self.device)
            loss[name_giou] = torch.tensor(0., device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain['bbox'] * F.l1_loss(pred_bboxes, gt_bboxes, reduction='sum') / num_gts
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        loss = {k: v.squeeze() for k, v in loss.items()}
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts, postfix=''):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = f'loss_mask{postfix}'
        name_dice = f'loss_dice{postfix}'

        loss = {}
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.tensor(0., device=self.device)
            loss[name_dice] = torch.tensor(0., device=self.device)
            return loss

        src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
        src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
        # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
        loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
                                                                        torch.tensor([num_gts], dtype=torch.float32))
        loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      pred_bboxes,
                      pred_scores,
                      gt_bboxes,
                      gt_cls,
                      gt_numgts,
                      num_gts,
                      dn_match_indices=None,
                      postfix='',
                      masks=None,
                      gt_mask=None):
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(pred_bboxes[self.uni_match_ind],
                                         pred_scores[self.uni_match_ind],
                                         gt_bboxes,
                                         gt_cls,
                                         gt_numgts,
                                         masks=masks[self.uni_match_ind] if masks is not None else None,
                                         gt_mask=gt_mask)
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(aux_bboxes,
                                             aux_scores,
                                             gt_bboxes,
                                             gt_cls,
                                             gt_numgts,
                                             masks=aux_masks,
                                             gt_mask=gt_mask)
            # TODO
            idx, gt_idx = self._get_index(match_indices)
            pred_bboxes_ = aux_bboxes[idx]
            # gt_bboxes_ = torch.cat([t[i] for t, (_, i) in zip(gt_bboxes, match_indices)], dim=0)
            gt_bboxes_ = gt_bboxes[gt_idx]
            iou_score = bbox_iou(pred_bboxes_.detach(), gt_bboxes_, xywh=True).squeeze(-1) \
                    if self.vfl and len(gt_bboxes) else None

            loss[0] += self._get_loss_class(
                aux_scores,
                gt_cls,
                match_indices,
                num_gts,
                postfix,
                iou_score,
            )[f'loss_class{postfix}']
            loss_ = self._get_loss_bbox(pred_bboxes_, gt_bboxes_, num_gts, postfix)
            loss[1] += loss_[f'loss_bbox{postfix}']
            loss[2] += loss_[f'loss_giou{postfix}']
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, num_gts, postfix)
                loss[3] += loss_[f'loss_mask{postfix}']
                loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f'loss_class_aux{postfix}': loss[0],
            f'loss_bbox_aux{postfix}': loss[1],
            f'loss_giou_aux{postfix}': loss[2]}
        if masks is not None and gt_mask is not None:
            loss[f'loss_mask_aux{postfix}'] = loss[3]
            loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    def _get_index(self, match_indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        pred_assigned = torch.cat([
            t[I] if len(I) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (I, _) in zip(pred_bboxes, match_indices)])
        gt_assigned = torch.cat([
            t[J] if len(J) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (_, J) in zip(gt_bboxes, match_indices)])
        return pred_assigned, gt_assigned

    def _get_prediction_loss(self,
                             pred_bboxes,
                             pred_scores,
                             gt_bboxes,
                             gt_cls,
                             gt_numgts,
                             masks=None,
                             gt_mask=None,
                             postfix='',
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(pred_bboxes,
                                         pred_scores,
                                         gt_bboxes,
                                         gt_cls,
                                         gt_numgts,
                                         masks=masks,
                                         gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        # TODO
        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes = pred_bboxes[idx]
        # gt_bboxes = torch.cat([t[i] for t, (_, i) in zip(gt_bboxes, match_indices)], dim=0)
        gt_bboxes = gt_bboxes[gt_idx]
        iou_score = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1) \
                if self.vfl and len(gt_bboxes) else None

        loss = {}
        loss.update(self._get_loss_class(pred_scores, gt_cls, match_indices, num_gts, postfix, iou_score))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, num_gts, postfix))
        if masks is not None and gt_mask is not None:
            loss.update(self._get_loss_mask(masks, gt_mask, match_indices, num_gts, postfix))
        return loss

    def forward(self, pred_bboxes, pred_scores, batch, masks=None, gt_mask=None, postfix='', **kwargs):
        """
        Args:
            pred_bboxes (Tensor): [l, b, query, 4]
            pred_scores (Tensor): [l, b, query, num_classes]
            gt_bboxes (List(Tensor)): list[[n, 4]]
            gt_cls (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """
        self.device = pred_bboxes.device

        dn_match_indices = kwargs.get('dn_match_indices', None)
        num_gts = kwargs.get('num_gts', None)

        gt_cls, gt_bboxes, gt_numgts = batch['cls'], batch['bboxes'], batch['num_gts']
        total_loss = self._get_prediction_loss(pred_bboxes[-1],
                                               pred_scores[-1],
                                               gt_bboxes,
                                               gt_cls,
                                               gt_numgts,
                                               masks=masks[-1] if masks is not None else None,
                                               gt_mask=gt_mask,
                                               postfix=postfix,
                                               dn_match_indices=dn_match_indices,
                                               num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(pred_bboxes[:-1],
                                   pred_scores[:-1],
                                   gt_bboxes,
                                   gt_cls,
                                   gt_numgts,
                                   num_gts,
                                   dn_match_indices,
                                   postfix,
                                   masks=masks[:-1] if masks is not None else None,
                                   gt_mask=gt_mask))

        return total_loss


class RTDETRDetectionLoss(DETRLoss):

    def forward(self, preds, batch, dn_out_bboxes=None, dn_out_logits=None, dn_meta=None):
        boxes, logits = preds
        num_gts = max(sum(batch['num_gts']), 1)
        total_loss = super().forward(boxes, logits, batch, num_gts=num_gts)

        if dn_meta is not None:
            dn_pos_idx, dn_num_group = \
                dn_meta['dn_pos_idx'], dn_meta['dn_num_group']
            assert len(batch['num_gts']) == len(dn_pos_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(batch['cls'], dn_pos_idx, dn_num_group, batch['num_gts'])

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super().forward(dn_out_bboxes,
                                      dn_out_logits,
                                      batch,
                                      postfix='_dn',
                                      dn_match_indices=dn_match_indices,
                                      num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update({f'{k}_dn': torch.tensor(0., device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_pos_idx, dn_num_group, gt_numgts):
        dn_match_indices = []
        labels = labels.split([n for n in gt_numgts])
        gt_numgts = [0] + gt_numgts[:-1]
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.int32) + gt_numgts[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), 'Expected the sa'
                f'me length, but got {len(dn_pos_idx[i])} and '
                f'{len(gt_idx)} respectively.'
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.int32), torch.zeros([0], dtype=torch.int32)))
        return dn_match_indices
