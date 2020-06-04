import tensorflow as tf

def mask2result(bboxes, masks, labels, meta, threshold=0.25, num_classes=81):
    '''
    Reformat masks from model output to be used for eval
    '''
    height, width = (bboxes[:,2]-bboxes[:,0], bboxes[:,3]-bboxes[:,1])
    sizes = tf.cast(tf.transpose(tf.stack([height, width])), tf.int32)
    img_size = meta[:2]
    if meta[-1]:
        masks = tf.reverse(masks, axis=[2])
    mask_list = []
    bboxes = tf.split(bboxes, 100)
    for idx in range(tf.shape(masks)[0]):
        mask = masks[idx]
        size = sizes[idx]
        bbox = bboxes[idx][0]
        bbox = tf.reshape(tf.transpose(tf.stack([tf.clip_by_value(bbox[0::2], 0, img_size[0]), 
                                         tf.clip_by_value(bbox[1::2], 0, img_size[1])])), [-1])
        if tf.math.multiply(*size)==0:
            mask_list.append(tf.cast(-tf.ones(tf.cast(img_size, tf.int32)), dtype=tf.int32))
        else:
            a_mask = tf.image.resize(mask, size)
            a_mask = tf.squeeze(a_mask>threshold)
            a_mask = tf.cast(a_mask, tf.int32)
            if a_mask.ndim!=2:
                a_mask = tf.expand_dims(a_mask, axis=0)
            a_mask = tf.pad(a_mask, [[tf.cast(bbox[0], tf.int32), 
                                      tf.cast(img_size[0]-bbox[2], tf.int32)],
                                     [tf.cast(bbox[1], tf.int32), 
                                      tf.cast(img_size[1]-bbox[3], tf.int32)]])
        mask_list.append(a_mask)
    
    mask_list_of_lists = [[]]*num_classes
    for mask, label in zip(mask_list, labels):
        mask_list_of_lists[label.numpy()].append(mask.numpy())
    return mask_list_of_lists
