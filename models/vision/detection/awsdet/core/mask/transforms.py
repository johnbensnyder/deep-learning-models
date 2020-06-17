import numpy as np
import cv2
from collections import defaultdict
import pycocotools.mask as mask_util
import tensorflow as tf

def box_clip(bboxes, img_size):
    y1, x1, y2, x2 = np.split(bboxes, 4, axis=1)
    cy1 = np.clip(y1, 0, img_size[0])
    cx1 = np.clip(x1, 0, img_size[1])
    cy2 = np.clip(y2, 0, img_size[0])
    cx2 = np.clip(x2, 0, img_size[1])
    clipped_boxes = np.transpose(np.squeeze(np.stack([cy1, cx1, cy2, cx2])))
    '''cheight = cy2 - cy1
    cwidth = cx2 - cx1
    crop_height_start = np.abs(y1)
    crop_width_start = np.abs(x1)'''
    return clipped_boxes

def compute_pads(bboxes, img_size):
    y1, x1, y2, x2 = np.split(bboxes, 4, axis=1)
    y1_pad = y1
    x1_pad = x1
    y2_pad = img_size[0] - y2
    x2_pad = img_size[1] - x2
    pads = np.transpose(np.squeeze(np.stack([y1_pad, x1_pad, y2_pad, x2_pad])))
    return pads

def fit_pad(mask, pad):
    mask = np.pad(mask, ((pad[0], pad[2]), (pad[1], pad[3])))
    return mask

def fit_mask_to_image(mask, cv2_size, clipped_box_size, 
                      clipped_boxes_np, pad, img_shape):
    if np.multiply(*clipped_box_size)==0:
        return np.zeros(img_shape, dtype=np.int32)
    mask = cv2.resize(mask, tuple(cv2_size), interpolation=cv2.INTER_NEAREST)
    if mask.ndim==1:
        mask = np.expand_dims(mask, axis=np.argmin(clipped_box_size))
    mask = mask[:clipped_box_size[0],:clipped_box_size[1]]
    #print('\n')
    #print(mask.shape)
    mask = fit_pad(mask, pad)
    #print(mask.shape)
    #print(clipped_boxes_np)
    return mask

def reshape_by_labels(mask_list, labels, num_classes=81):
    list_of_lists = [[]]*num_classes
    for mask, label in zip(mask_list, labels):
        list_of_lists[label].append(mask)
    return list_of_lists

def mask2result(masks, labels, meta, num_classes=81, threshold=0.5):
    meta = np.squeeze(meta)
    img_heights, img_widths = meta[:2].astype(np.int32)
    unpadded_height = tf.cast(meta[3], tf.int32)
    unpadded_width = tf.cast(meta[4], tf.int32)
    orig_height = tf.cast(meta[0], tf.int32)
    orig_width = tf.cast(meta[1], tf.int32)
    masks = masks[:,:unpadded_height,:unpadded_width, :]
    masks = tf.image.resize(masks, (orig_height, orig_width), method='nearest')
    masks_np = np.squeeze((masks.numpy()>threshold).astype(np.int32))
    labels_np = labels.numpy()
    if meta[-1]==1:
        masks_np = np.flip(masks_np, axis=2)
    lists = defaultdict(list)
    for i,j in enumerate(labels_np):
        lists[j].append(mask_util.encode(
                    np.array(
                        masks_np[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])
    return lists

'''def mask2result(bboxes, masks, labels, meta, num_classes=81, threshold=0.5):
    # convert tensors to numpy
    # round bboxes to nearest int
    meta = np.squeeze(meta)
    img_heights, img_widths = meta[:2].astype(np.int32)
    bboxes_np = np.round(bboxes).astype(np.int32)
    clipped_boxes_np = box_clip(bboxes_np, (img_heights, img_widths))
    masks_np = (masks.numpy()>threshold).astype(np.int32)
    labels_np = labels.numpy()
    if meta[-1]==1:
        masks_np = np.flip(masks_np, axis=2)
    bbox_heights = bboxes_np[:,2]-bboxes_np[:,0]
    bbox_clipped_heights = clipped_boxes_np[:,2]-clipped_boxes_np[:,0]
    bbox_widths = bboxes_np[:,3]-bboxes_np[:,1]
    bbox_clipped_widths = clipped_boxes_np[:,3]-clipped_boxes_np[:,1]
    bbox_sizes = np.transpose(np.stack([bbox_heights, bbox_widths]))
    #cv2 needs dims in opposite direction
    cv2_sizes = np.flip(bbox_sizes, axis=1)
    bbox_clipped_sizes = np.squeeze(np.transpose([np.stack([bbox_clipped_heights,
                                                 bbox_clipped_widths])]))
    pads = compute_pads(clipped_boxes_np, (img_heights, img_widths))
    
    mask_list = []
    for idx in range(100):
        #print(idx)
        mask_list.append(fit_mask_to_image(masks_np[idx],
                                           cv2_sizes[idx],
                                           bbox_clipped_sizes[idx],
                                           clipped_boxes_np[idx],
                                           pads[idx],
                                           (img_heights, img_widths)))
    lists = defaultdict(list)
    for i,j in enumerate(labels_np):
        lists[j].append(mask_util.encode(
                    np.array(
                        mask_list[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])
    return lists'''