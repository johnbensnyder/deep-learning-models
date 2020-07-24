from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np
import pycocotools.mask as maskUtils
from awsdet.utils.image.geometric import rescale_size, imrescale, imflip, impad

class BaseInstanceMasks(metaclass=ABCMeta):
    """Base class for instance masks."""

    @abstractmethod
    def rescale(self, scale, interpolation='nearest'):
        """Rescale masks as large as possible while keeping the aspect ratio.
        For details can refer to `mmcv.imrescale`.
        Args:
            scale (tuple[int]): The maximum size (h, w) of rescaled mask.
            interpolation (str): Same as :func:`mmcv.imrescale`.
        Returns:
            BaseInstanceMasks: The rescaled masks.
        """
        pass

    @abstractmethod
    def resize(self, out_shape, interpolation='nearest'):
        """Resize masks to the given out_shape.
        Args:
            out_shape: Target (h, w) of resized mask.
            interpolation (str): See :func:`mmcv.imresize`.
        Returns:
            BaseInstanceMasks: The resized masks.
        """
        pass

    @abstractmethod
    def flip(self, flip_direction='horizontal'):
        """Flip masks alone the given direction.
        Args:
            flip_direction (str): Either 'horizontal' or 'vertical'.
        Returns:
            BaseInstanceMasks: The flipped masks.
        """
        pass

    @abstractmethod
    def pad(self, out_shape, pad_val):
        """Pad masks to the given size of (h, w).
        Args:
            out_shape (tuple[int]): Target (h, w) of padded mask.
            pad_val (int): The padded value.
        Returns:
            BaseInstanceMasks: The padded masks.
        """
        pass

    @abstractmethod
    def crop(self, bbox):
        """Crop each mask by the given bbox.
        Args:
            bbox (ndarray): Bbox in format [x1, y1, x2, y2], shape (4, ).
        Return:
            BaseInstanceMasks: The cropped masks.
        """
        pass

    @abstractmethod
    def expand(self, expanded_h, expanded_w, top, left):
        """see :class:`Expand`."""
        pass

    @property
    @abstractmethod
    def areas(self):
        """ndarray: areas of each instance."""
        pass

    @abstractmethod
    def to_ndarray(self):
        """Convert masks to the format of ndarray.
        Return:
            ndarray: Converted masks in the format of ndarray.
        """
        pass

    @abstractmethod
    def to_tensor(self, dtype, device):
        """Convert masks to the format of Tensor.
        Args:
            dtype (str): Dtype of converted mask.
            device (torch.device): Device of converted masks.
        Returns:
            Tensor: Converted masks in the format of Tensor.
        """
        pass
    
class BitmapMasks(BaseInstanceMasks):
    """This class represents masks in the form of bitmaps.
    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks
    """

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width
            
    def __getitem__(self, index):
        """Index the BitmapMask.
        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.
        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)
    
    def __iter__(self):
        return iter(self.masks)
    
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)
    
    def rescale(self, scale, interpolation='nearest'):
        """See :func:`BaseInstanceMasks.rescale`."""
        if len(self.masks) == 0:
            new_w, new_h = rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack([
                imrescale(mask, scale, interpolation=interpolation)
                for mask in self.masks
            ])
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)
    
    def resize(self, out_shape, interpolation='nearest'):
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack([
                imresize(mask, out_shape, interpolation=interpolation)
                for mask in self.masks
            ])
        return BitmapMasks(resized_masks, *out_shape)
    
    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical')

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack([
                imflip(mask, direction=flip_direction)
                for mask in self.masks
            ])
        return BitmapMasks(flipped_masks, self.height, self.width)
    
    def pad(self, out_shape, pad_val=0):
        """See :func:`BaseInstanceMasks.pad`."""
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack([
                impad(mask, shape=out_shape, pad_val=pad_val)
                for mask in self.masks
            ])
        return BitmapMasks(padded_masks, *out_shape)
    
    def crop(self, bbox):
        """See :func:`BaseInstanceMasks.crop`."""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1:y1 + h, x1:x1 + w]
        return BitmapMasks(cropped_masks, h, w)
    
    def expand(self, expanded_h, expanded_w, top, left):
        """See :func:`BaseInstanceMasks.expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w),
                                     dtype=np.uint8)
        else:
            expanded_mask = np.zeros((len(self), expanded_h, expanded_w),
                                     dtype=np.uint8)
            expanded_mask[:, top:top + self.height,
                          left:left + self.width] = self.masks
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)
    
    @property
    def areas(self):
        """See :py:attr:`BaseInstanceMasks.areas`."""
        return self.masks.sum((1, 2))

    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks

    def to_tensor(self, dtype=tf.int32):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        return tf.convert_to_tensor(self.masks, dtype=dtype)
    
class PolygonMasks(BaseInstanceMasks):
    """This class represents masks in the form of polygons.
    Polygons is a list of three levels. The first level of the list
    corresponds to objects, the second level to the polys that compose the
    object, the third level to the poly coordinates
    Args:
        masks (list[list[ndarray]]): The first level of the list
            corresponds to objects, the second level to the polys that
            compose the object, the third level to the poly coordinates
        height (int): height of masks
        width (int): width of masks
    """

    def __init__(self, masks, height, width):
        assert isinstance(masks, list)
        if len(masks) > 0:
            assert isinstance(masks[0], list)
            assert isinstance(masks[0][0], np.ndarray)

        self.height = height
        self.width = width
        self.masks = masks
        
    def __getitem__(self, index):
        """Index the polygon masks.
        Args:
            index (ndarray | List): The indices.
        Returns:
            :obj:`PolygonMasks`: The indexed polygon masks.
        """
        if isinstance(index, np.ndarray):
            index = index.tolist()
        if isinstance(index, list):
            masks = [self.masks[i] for i in index]
        else:
            try:
                masks = self.masks[index]
            except Exception:
                raise ValueError(
                    f'Unsupported input of type {type(index)} for indexing!')
        if isinstance(masks[0], np.ndarray):
            masks = [masks]  # ensure a list of three levels
        return PolygonMasks(masks, self.height, self.width)
    
    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation=None):
        """see :func:`BaseInstanceMasks.rescale`"""
        new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
        if len(self.masks) == 0:
            rescaled_masks = PolygonMasks([], new_h, new_w)
        else:
            rescaled_masks = self.resize((new_h, new_w))
        return rescaled_masks

    def resize(self, out_shape, interpolation=None):
        """see :func:`BaseInstanceMasks.resize`"""
        if len(self.masks) == 0:
            resized_masks = PolygonMasks([], *out_shape)
        else:
            h_scale = out_shape[0] / self.height
            w_scale = out_shape[1] / self.width
            resized_masks = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] *= w_scale
                    p[1::2] *= h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            resized_masks = PolygonMasks(resized_masks, *out_shape)
        return resized_masks

    def flip(self, flip_direction='horizontal'):
        """see :func:`BaseInstanceMasks.flip`"""
        assert flip_direction in ('horizontal', 'vertical')
        if len(self.masks) == 0:
            flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            if flip_direction == 'horizontal':
                dim = self.width
                idx = 0
            else:
                dim = self.height
                idx = 1
            flipped_masks = []
            for poly_per_obj in self.masks:
                flipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[idx::2] = dim - p[idx::2]
                    flipped_poly_per_obj.append(p)
                flipped_masks.append(flipped_poly_per_obj)
            flipped_masks = PolygonMasks(flipped_masks, self.height,
                                         self.width)
        return flipped_masks

    def crop(self, bbox):
        """see :func:`BaseInstanceMasks.crop`"""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = PolygonMasks([], h, w)
        else:
            cropped_masks = []
            for poly_per_obj in self.masks:
                cropped_poly_per_obj = []
                for p in poly_per_obj:
                    # pycocotools will clip the boundary
                    p = p.copy()
                    p[0::2] -= bbox[0]
                    p[1::2] -= bbox[1]
                    cropped_poly_per_obj.append(p)
                cropped_masks.append(cropped_poly_per_obj)
            cropped_masks = PolygonMasks(cropped_masks, h, w)
        return cropped_masks

    def pad(self, out_shape, pad_val=0):
        """padding has no effect on polygons`"""
        return PolygonMasks(self.masks, *out_shape)

    def expand(self, *args, **kwargs):
        """TODO: Add expand for polygon"""
        raise NotImplementedError
        
    def to_bitmap(self):
        """convert polygon masks to bitmap masks."""
        bitmap_masks = self.to_ndarray()
        return BitmapMasks(bitmap_masks, self.height, self.width)

    @property
    def areas(self):
        """Compute areas of masks.
        This func is modified from `detectron2
        <https://github.com/facebookresearch/detectron2/blob/ffff8acc35ea88ad1cb1806ab0f00b4c1c5dbfd9/detectron2/structures/masks.py#L387>`_.
        The function only works with Polygons using the shoelace formula.
        Return:
            ndarray: areas of each instance
        """  # noqa: W501
        area = []
        for polygons_per_obj in self.masks:
            area_per_obj = 0
            for p in polygons_per_obj:
                area_per_obj += self._polygon_area(p[0::2], p[1::2])
            area.append(area_per_obj)
        return np.asarray(area)

    def _polygon_area(self, x, y):
        """Compute the area of a component of a polygon.
        Using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        Args:
            x (ndarray): x coordinates of the component
            y (ndarray): y coordinates of the component
        Return:
            float: the are of the component
        """  # noqa: 501
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def to_ndarray(self):
        """Convert masks to the format of ndarray."""
        if len(self.masks) == 0:
            return np.empty((0, self.height, self.width), dtype=np.uint8)
        bitmap_masks = []
        for poly_per_obj in self.masks:
            bitmap_masks.append(
                polygon_to_bitmap(poly_per_obj, self.height, self.width))
        return np.stack(bitmap_masks)
    
    def to_tensor(self, dtype=tf.int32):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        if len(self.masks) == 0:
            return tf.zeros((self.height, self.width),
                               dtype=dtype)
        ndarray_masks = self.to_ndarray()
        return tf.convert_to_tensor(ndarray_masks, dtype=dtype)
    
def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.
    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width
    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(np.bool)
    return bitmap_mask
