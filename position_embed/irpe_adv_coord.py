
"""The implementation of iRPE (image relative position encoding)."""
from easydict import EasyDict as edict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def piecewise_index(adv_shift,relative_position, alpha, beta, gamma):
    """piecewise index function defined in Eq. (18) in our paper.

    Parameters
    ----------
    adv_shift: torch.Tensor
        shape 1, indicates the shifting
    relative_position: torch.Tensor, dtype: long or float
        The shape of `relative_position` is (L, L).
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    idx: torch.Tensor, dtype: long
        A tensor indexing relative distances to corresponding encodings.
        `idx` is a long tensor, whose shape is (L, L) and each element is in [-beta, beta].
    """
    relative_position = relative_position+adv_shift
    with torch.no_grad():
        rp_abs = relative_position.abs()
        mask = rp_abs <= alpha #direct use y=x
        not_mask = ~mask
    #outside for indexing functions
    rp_out = relative_position[not_mask]
    #rp_abs_out = rp_abs[not_mask]
    #rp_out = relative_position.clone()
    y_out = (torch.sign(rp_out) * (alpha +torch.log(rp_out.abs() / alpha) /math.log(gamma / alpha) *(beta - alpha)).clip(max=beta))
    idx = relative_position.float()

    idx[not_mask] = y_out
    #idx = torch.where(rp_abs<=alpha,relative_position,y_out)
    return idx
# @torch.no_grad()
# def piecewise_index(relative_position, alpha, beta, gamma, dtype):
#     """piecewise index function defined in Eq. (18) in our paper.
#
#     Parameters
#     ----------
#     relative_position: torch.Tensor, dtype: long or float
#         The shape of `relative_position` is (L, L).
#     alpha, beta, gamma: float
#         The coefficients of piecewise index function.
#
#     Returns
#     -------
#     idx: torch.Tensor, dtype: long
#         A tensor indexing relative distances to corresponding encodings.
#         `idx` is a long tensor, whose shape is (L, L) and each element is in [-beta, beta].
#     """
#     rp_abs = relative_position.abs()
#     mask = rp_abs <= alpha
#     not_mask = ~mask
#     rp_out = relative_position[not_mask]
#     rp_abs_out = rp_abs[not_mask]
#     y_out = (torch.sign(rp_out) * (alpha +
#                                    torch.log(rp_abs_out / alpha) /
#                                    math.log(gamma / alpha) *
#                                    (beta - alpha)).round().clip(max=beta)).to(dtype)
#
#     idx = relative_position.clone()
#     if idx.dtype in [torch.float32, torch.float64]:
#         # round(x) when |x| <= alpha
#         idx = idx.round().to(dtype)
#
#     # assign the value when |x| > alpha
#     idx[not_mask] = y_out
#     return idx


def get_absolute_positions(height, width, dtype=torch.long, device=torch.device('cpu')):
    '''Get absolute positions

    Take height = 3, width = 3 as an example:
    rows:        cols:
    1 1 1        1 2 3
    2 2 2        1 2 3
    3 3 3        1 2 3

    return stack([rows, cols], 2)

    Parameters
    ----------
    height, width: int
        The height and width of feature map
    dtype: torch.dtype
        the data type of returned value
    device: torch.device
        the device of returned value

    Return
    ------
    2D absolute positions: torch.Tensor
        The shape is (height, width, 2),
        where 2 represents a 2D position (row, col).
    '''
    rows = torch.arange(height, dtype=dtype, device=device).view(
        height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=dtype, device=device).view(
        1, width).repeat(height, 1)
    return torch.stack([rows, cols], 2)


@torch.no_grad()
def quantize_values(values):
    """Quantization: Map all values (long or float) into a discrte integer set.

    Parameters
    ----------
    values: torch.Tensor, dtype: long or float
        arbitrary shape

    Returns
    -------
    res: torch.Tensor, dtype: long
        The quantization result starts at 0.
        The shape is the same as that of `values`.
    uq.numel(): long
        The number of the quantization integers, namely `res` is in [0, uq.numel()).
    """
    # quantize and re-assign bucket id
    res = torch.empty_like(values)
    uq = values.unique()
    cnt = 0
    for (tid, v) in enumerate(uq):
        mask = (values == v)
        cnt += torch.count_nonzero(mask)
        res[mask] = tid
    assert cnt == values.numel()
    return res, uq.numel()


class METHOD:
    """define iRPE method IDs
    We divide the implementation of CROSS into CROSS_ROWS and CROSS_COLS.

    """
    EUCLIDEAN = 0
    QUANT = 1
    PRODUCT = 3
    CROSS = 4
    CROSS_ROWS = 41
    CROSS_COLS = 42








def _rp_2d_product( adv_shift,diff, **kwargs):
    """2D RPE with Product method.

    Parameters
    ----------
    adv_shift: torch.Tensor
        shape 2, indicates x shifting and y shifting

    diff: torch.Tensor
        The shape of `diff` is (L, L, 2), #calculated from L*L coord matrix
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    # convert beta to an integer since beta is a float number.
    beta = kwargs['beta']

    #S = 2 * beta_int + 1 #num buckets_row
    # the output of piecewise index function is in [-beta_int, beta_int]
    #r = piecewise_index(adv_shift[0],diff[:, :, 0], **kwargs) + \
    #    beta  # [0, 2 * beta_int]
    #L*L shape, row_id
    #c = piecewise_index(adv_shift[1],diff[:, :, 1], **kwargs) + \
    #    beta  # [0, 2 * beta_int]
    #adv_shift = adv_shift.unsqueeze(0).unsqueeze(0)
    adv_shift = adv_shift.unsqueeze(0).unsqueeze(0)
    pid = piecewise_index(adv_shift,diff, **kwargs) #+beta, no shift directly normalized to [-1,1]
    #L*L shape, column shape
    #pid = r * S+ c
    return pid




# Define a mapping from METHOD_ID to Python function
_METHOD_FUNC = {
    METHOD.PRODUCT: _rp_2d_product,
}


def get_num_buckets(method, alpha, beta, gamma):
    """ Get number of buckets storing relative position encoding.
    The buckets does not contain `skip` token.

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    num_buckets: int
        The number of buckets storing relative position encoding.
    """
    beta_int = int(beta)
    if method == METHOD.PRODUCT:
        # IDs in [0, (2 * beta_int + 1)^2) for Product method
        num_buckets = (2 * beta_int + 1) ** 2
    else:
        # IDs in [-beta_int, beta_int] except of Product method
        num_buckets = 2 * beta_int + 1
    return num_buckets


# (method, alpha, beta, gamma) -> (bucket_ids, num_buckets, height, width)
BUCKET_IDS_BUF = dict()

def get_bucket_ids_2d_without_skip(adv_shift,method, height, width,
                                   alpha, beta, gamma,
                                 device=torch.device('cpu')):
    """Get bucket IDs for image relative position encodings without skip token

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    height, width: int
        The height and width of the feature map.
        The sequence length is equal to `height * width`.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.
    dtype: torch.dtype
        the data type of returned `bucket_ids`
    device: torch.device
        the device of returned `bucket_ids`

    Returns
    -------
    bucket_ids: torch.Tensor, dtype: long
        The bucket IDs which index to corresponding encodings.
        The shape of `bucket_ids` is (skip + L, skip + L),
        where `L = height * wdith`.
    num_buckets: int
        The number of buckets including `skip` token.
    L: int
        The sequence length
    """
    key = (method, alpha, beta, gamma,  device)
    max_height, max_width = height, width
    func = _METHOD_FUNC.get(method, None)
    if func is None:
        raise NotImplementedError(
            f"[Error] The method ID {method} does not exist.")
    pos = get_absolute_positions(max_height, max_width,device=device)
    # compute the offset of a pair of 2D relative positions
    max_L = max_height * max_width
    pos1 = pos.view((max_L, 1, 2))
    pos2 = pos.view((1, max_L, 2))
    # diff: shape of (L, L, 2)
    diff = pos1 - pos2
    # bucket_ids: shape of (L, L)

    #L*L*2
    bucket_ids = func(adv_shift,diff, alpha=alpha, beta=beta,
                          gamma=gamma)
    beta_int = int(beta)

    num_buckets = get_num_buckets(method, alpha, beta, gamma)
    L = height * width
    return bucket_ids, num_buckets, L
# @torch.no_grad()
# def get_bucket_ids_2d_without_skip(method, height, width,
#                                    alpha, beta, gamma,
#                                    dtype=torch.long, device=torch.device('cpu')):
#     """Get bucket IDs for image relative position encodings without skip token
#
#     Parameters
#     ----------
#     method: METHOD
#         The method ID of image relative position encoding.
#     height, width: int
#         The height and width of the feature map.
#         The sequence length is equal to `height * width`.
#     alpha, beta, gamma: float
#         The coefficients of piecewise index function.
#     dtype: torch.dtype
#         the data type of returned `bucket_ids`
#     device: torch.device
#         the device of returned `bucket_ids`
#
#     Returns
#     -------
#     bucket_ids: torch.Tensor, dtype: long
#         The bucket IDs which index to corresponding encodings.
#         The shape of `bucket_ids` is (skip + L, skip + L),
#         where `L = height * wdith`.
#     num_buckets: int
#         The number of buckets including `skip` token.
#     L: int
#         The sequence length
#     """
#
#     key = (method, alpha, beta, gamma, dtype, device)
#     value = BUCKET_IDS_BUF.get(key, None)
#     if value is None or value[-2] < height or value[-1] < width:
#         if value is None:
#             max_height, max_width = height, width
#         else:
#             max_height = max(value[-2], height)
#             max_width = max(value[-1], width)
#         # relative position encoding mapping function
#         func = _METHOD_FUNC.get(method, None)
#         if func is None:
#             raise NotImplementedError(
#                 f"[Error] The method ID {method} does not exist.")
#         pos = get_absolute_positions(max_height, max_width, dtype, device)
#
#         # compute the offset of a pair of 2D relative positions
#         max_L = max_height * max_width
#         pos1 = pos.view((max_L, 1, 2))
#         pos2 = pos.view((1, max_L, 2))
#         # diff: shape of (L, L, 2)
#         diff = pos1 - pos2
#
#         # bucket_ids: shape of (L, L)
#         bucket_ids = func(diff, alpha=alpha, beta=beta,
#                           gamma=gamma, dtype=dtype)
#         beta_int = int(beta)
#         if method != METHOD.PRODUCT:
#             bucket_ids += beta_int
#         bucket_ids = bucket_ids.view(
#             max_height, max_width, max_height, max_width)
#
#         num_buckets = get_num_buckets(method, alpha, beta, gamma)
#         value = (bucket_ids, num_buckets, height, width)
#         BUCKET_IDS_BUF[key] = value
#     L = height * width
#     bucket_ids = value[0][:height, :width, :height, :width].reshape(L, L)
#     num_buckets = value[1]
#
#     return bucket_ids, num_buckets, L
def get_bucket_ids_2d(adv_shift,method, height, width,
                      skip, alpha, beta, gamma,
                   device=torch.device('cpu')):
    """Get bucket IDs for image relative position encodings

    Parameters
    ----------
    method: METHOD
        The method ID of image relative position encoding.
    height, width: int
        The height and width of the feature map.
        The sequence length is equal to `height * width`.
    skip: int [0 or 1]
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
    alpha, beta, gamma: float
        The coefficients of piecewise index function.
    dtype: torch.dtype
        the data type of returned `bucket_ids`
    device: torch.device
        the device of returned `bucket_ids`

    Returns
    -------
    bucket_ids: torch.Tensor, dtype: long
        The bucket IDs which index to corresponding encodings.
        The shape of `bucket_ids` is (skip + L, skip + L),
        where `L = height * wdith`.
    num_buckets: int
        The number of buckets including `skip` token.
    """
    assert skip in [
        0, 1], f"`get_bucket_ids_2d` only support skip is 0 or 1, current skip={skip}"
    # bucket_id_row,bucket_id_column, num_buckets, L = get_bucket_ids_2d_without_skip(adv_shift,
    #                                                             method, height, width,
    #                                                             alpha, beta, gamma,
    #                                                              device)

    bucket_ids, num_buckets, L = get_bucket_ids_2d_without_skip(adv_shift,method, height, width,
                                                                alpha, beta, gamma,
                                                                 device)
    # add an extra encoding (id = num_buckets) for the classification token
    # if skip > 0:#skip 1 for support class token
    #     assert skip == 1, "`get_bucket_ids_2d` only support skip is 0 or 1"
    #     new_bids = bucket_ids.new_empty(size=(skip + L, skip + L,2))
    #     new_bids[0] = num_buckets
    #     new_bids[:, 0] = num_buckets
    #     new_bids[skip:, skip:] = bucket_ids
    #     bucket_ids = new_bids
        # new_bids = bucket_id_row.new_empty(size=(skip + L, skip + L))
        # new_bids[0] = num_buckets
        # new_bids[:, 0] = num_buckets
        # new_bids[skip:, skip:] = bucket_id_row
        # bucket_id_row = new_bids
        # new_bids2 = bucket_id_column.new_empty(size=(skip + L, skip + L))
        # new_bids2[0] = num_buckets
        # new_bids2[:, 0] = num_buckets
        # new_bids2[skip:, skip:] = bucket_id_column
        # bucket_id_column = new_bids2
    bucket_ids = bucket_ids.contiguous()
    # bucket_id_row = bucket_id_row.contiguous()
    # bucket_id_column = bucket_id_column.contiguous()
    return bucket_ids, num_buckets + skip

# @torch.no_grad()
# def get_bucket_ids_2d(method, height, width,
#                       skip, alpha, beta, gamma,
#                       dtype=torch.long, device=torch.device('cpu')):
#     """Get bucket IDs for image relative position encodings
#
#     Parameters
#     ----------
#     method: METHOD
#         The method ID of image relative position encoding.
#     height, width: int
#         The height and width of the feature map.
#         The sequence length is equal to `height * width`.
#     skip: int [0 or 1]
#         The number of skip token before spatial tokens.
#         When skip is 0, no classification token.
#         When skip is 1, there is a classification token before spatial tokens.
#     alpha, beta, gamma: float
#         The coefficients of piecewise index function.
#     dtype: torch.dtype
#         the data type of returned `bucket_ids`
#     device: torch.device
#         the device of returned `bucket_ids`
#
#     Returns
#     -------
#     bucket_ids: torch.Tensor, dtype: long
#         The bucket IDs which index to corresponding encodings.
#         The shape of `bucket_ids` is (skip + L, skip + L),
#         where `L = height * wdith`.
#     num_buckets: int
#         The number of buckets including `skip` token.
#     """
#     assert skip in [
#         0, 1], f"`get_bucket_ids_2d` only support skip is 0 or 1, current skip={skip}"
#     bucket_ids, num_buckets, L = get_bucket_ids_2d_without_skip(method, height, width,
#                                                                 alpha, beta, gamma,
#                                                                 dtype, device)
#
#     # add an extra encoding (id = num_buckets) for the classification token
#     if skip > 0:
#         assert skip == 1, "`get_bucket_ids_2d` only support skip is 0 or 1"
#         new_bids = bucket_ids.new_empty(size=(skip + L, skip + L))
#         new_bids[0] = num_buckets
#         new_bids[:, 0] = num_buckets
#         new_bids[skip:, skip:] = bucket_ids
#         bucket_ids = new_bids
#     bucket_ids = bucket_ids.contiguous()
#     return bucket_ids, num_buckets + skip

def point_sample(input, points, align_corners=False, **kwargs):
    """A wrapper around :function:`grid_sample` to support 3D point_coords
    tensors Unlike :function:`torch.nn.functional.grid_sample` it assumes
    point_coords to lie inside [0, 1] x [0, 1] square.

    Args:
        input (Tensor): Feature map, shape (N, C, H, W).
        points (Tensor): Image based absolute point coordinates (normalized),
            range [-1, 1] x [-1, 1], shape (N, P, 2) or (N, Hgrid, Wgrid, 2).
        align_corners (bool): Whether align_corners. Default: False

    Returns:
        Tensor: Features of `point` on `input`, shape (N, C, P) or
            (N, C, Hgrid, Wgrid).
    """

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, points, align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output
class iRPE(nn.Module):
    """The implementation of image relative position encoding (excluding Cross method).

    Parameters
    ----------
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    transposed: bool
        Whether to transpose the input feature.
        For iRPE on queries or keys, transposed should be `True`.
        For iRPE on values, transposed should be `False`.
    num_buckets: int
        The number of buckets, which store encodings.
    initializer: None or an inplace function
        [Optional] The initializer to `lookup_table`.
        Initalize `lookup_table` as zero by default.
    rpe_config: RPEConfig
        The config generated by the function `get_single_rpe_config`.
    """
    # a buffer to store bucket index
    # (key, rp_bucket, _ctx_rp_bucket_flatten)
    _rp_bucket_buf = (None, None, None)

    def __init__(self, head_dim, num_heads=8,
                 mode=None, method=None,
                 transposed=True, num_buckets=None,
                 initializer=None, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # relative position
        assert mode in [None, 'bias', 'contextual']
        self.mode = mode

        assert method is not None, 'method should be a METHOD ID rather than None'
        self.method = method

        self.transposed = transposed
        self.num_buckets = num_buckets

        if initializer is None:
            def initializer(x): return None
        self.initializer = initializer

        self.reset_parameters()

        self.rpe_config = rpe_config
        self.stable_rp_bucket = None
        self.adv_shift = nn.Parameter(torch.zeros(2))
        self.no_shift = nn.Parameter(torch.zeros(2),requires_grad=False)#used for default rpe



    @torch.no_grad()
    def reset_parameters(self):
        # initialize the parameters of iRPE
        if self.transposed:
            if self.mode == 'bias':
                self.lookup_table_bias = nn.Parameter(
                    torch.zeros(self.num_heads, self.num_buckets))
                self.initializer(self.lookup_table_bias)
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.head_dim, self.num_buckets))
                self.initializer(self.lookup_table_weight)
        else:
            if self.mode == 'bias':
                raise NotImplementedError(
                    "[Error] Bias non-transposed RPE does not exist.")
            elif self.mode == 'contextual':
                self.lookup_table_weight = nn.Parameter(
                    torch.zeros(self.num_heads,
                                self.num_buckets, self.head_dim))
                self.initializer(self.lookup_table_weight)

    def forward(self, x, mask,height=None, width=None):
        """forward function for iRPE.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head

        Returns
        -------
        rpe_encoding: torch.Tensor
            image Relative Position Encoding,
            whose shape is (B, H, L, L)
        """
        rp_buckets_mask = self._get_rp_bucket(x, adv_shift=True,height=height, width=width)
        rp_buckets_nonmask = self._get_rp_bucket(x, adv_shift=False,height=height, width=width)
        # rp_bucket, self._ctx_rp_bucket_flatten = \
        #     self._get_rp_bucket(x, height=height, width=width)
        # if self.transposed:
        return self.forward_rpe_transpose(x, rp_buckets_mask,rp_buckets_nonmask,mask)
        #return self.forward_rpe_no_transpose(x, rp_bucket)
    def _get_rp_bucket(self, x,adv_shift, height=None, width=None):
        """Get relative position encoding buckets IDs corresponding the input shape

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        height: int or None
            [Optional] The height of the input
            If not defined, height = floor(sqrt(L))
        width: int or None
            [Optional] The width of the input
            If not defined, width = floor(sqrt(L))

        Returns
        -------
        rp_bucket: torch.Tensor
            relative position encoding buckets IDs
            The shape is (L, L)
        _ctx_rp_bucket_flatten: torch.Tensor or None
            It is a private tensor for efficient computation.
        """
        # B, H, L, D = x.shape
        # device = x.device
        # if height is None:
        #     E = int(math.sqrt(L))#L is num_patches:196, small crop:49
        #     height = width = E#14,14; 7,7 for small crop
        # key = (height, width, device)
        # if self.stable_rp_bucket is not None and self._rp_bucket_buf==key:
        #     return self.stable_rp_bucket
        # # no buffer is allowed because adv coordinates
        # #skip = L - height * width
        # config = self.rpe_config
        #
        #
        # # rp_bucket_row,rp_bucket_column, num_buckets = get_bucket_ids_2d(adv_shift=self.adv_shift,method=self.method,
        # rp_buckets, num_buckets = get_bucket_ids_2d(method=self.method,
        #                                            height=height, width=width,
        #                                            skip=0, alpha=config.alpha,
        #                                            beta=config.beta, gamma=config.gamma,
        #                                             device=device)
        # #assert num_buckets == self.num_buckets
        #
        #
        # self._rp_bucket_buf=key
        # self.stable_rp_bucket=rp_buckets
        # return rp_buckets#, _ctx_rp_bucket_flatten
        B, H, L, D = x.shape
        device = x.device
        if height is None:
            E = int(math.sqrt(L))#L is num_patches:196, small crop:49
            height = width = E#14,14; 7,7 for small crop
        key = (height, width, device)
        if adv_shift is False and self.stable_rp_bucket is not None and self._rp_bucket_buf==key:
            return self.stable_rp_bucket
        # no buffer is allowed because adv coordinates
        skip = L - height * width
        config = self.rpe_config


        # rp_bucket_row,rp_bucket_column, num_buckets = get_bucket_ids_2d(adv_shift=self.adv_shift,method=self.method,
        if adv_shift:
            rp_buckets, num_buckets = get_bucket_ids_2d(adv_shift=self.adv_shift,method=self.method,
                                                   height=height, width=width,
                                                   skip=0, alpha=config.alpha,
                                                   beta=config.beta, gamma=config.gamma,
                                                    device=device)
        else:
            rp_buckets, num_buckets = get_bucket_ids_2d(adv_shift=self.no_shift,method=self.method,
                                                   height=height, width=width,
                                                   skip=0, alpha=config.alpha,
                                                   beta=config.beta, gamma=config.gamma,
                                                    device=device)
        #assert num_buckets == self.num_buckets


        if not adv_shift:
            self._rp_bucket_buf=key
            self.stable_rp_bucket=rp_buckets
        return rp_buckets#, _ctx_rp_bucket_flatten


    def forward_rpe_transpose(self, x, rp_buckets_mask,rp_buckets_nonmask,mask):
        """Forward function for iRPE (transposed version)
        This version is utilized by RPE on Query or Key

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        rp_bucket: torch.Tensor
            relative position encoding buckets IDs
            The shape is (L, L)

        Weights
        -------
        lookup_table_bias: torch.Tensor
            The shape is (H or 1, num_buckets)

        or

        lookup_table_weight: torch.Tensor
            The shape is (H or 1, head_dim, num_buckets)

        Returns
        -------
        output: torch.Tensor
            Relative position encoding on queries or keys.
            The shape is (B or 1, H, L, L),
            where D is the output dimension for each head.
        """

        B = len(x)  # batch_size

        B, H, L, head_dim = x.shape
        # H=1 for rij
        rij= self.lookup_table_weight#our r_ij in shape of H, H_C, (2beta+1)^2 , here the last dim is num of buckets that we will use.
        #(N, C, H, W)
        r_h,r_hdim,total_r_bucket = rij.shape
        #last bucket is for the class token
        num_buckets_row = int(2*int(self.rpe_config.beta)+1)
        rij_cls_token = rij[:,:,-1:]
        rij_reshape = rij[:,:,:-1].reshape(r_h,r_hdim,num_buckets_row,num_buckets_row)
        # 1, 64,7,7
        rij_reshape = rij_reshape.reshape(-1,num_buckets_row,num_buckets_row).unsqueeze(0)# 1,H*H_dim, num_bucket_row, num_bucket_column

        #point sample call
        #input (Tensor): Feature map, shape (N, C, H, W).
        #points (Tensor): Image based absolute point coordinates (normalized),
        #    range [-1, 1] x [-1, 1], shape (N, P, 2) or (N, Hgrid, Wgrid, 2).
        # Tensor: Features of `point` on `input`, shape (N, C, P)
        # normalize rp_buckets first
        rp_buckets_mask = torch.clamp(rp_buckets_mask, min=-self.rpe_config.beta, max=self.rpe_config.beta)
        rp_buckets_mask = rp_buckets_mask/self.rpe_config.beta
        L_bucket_row, L_bucket_column, _ = rp_buckets_mask.shape
        rp_buckets_mask = rp_buckets_mask.unsqueeze(0)# L_bucket_row, L_bucket_column, 2
        #rij_reshape = rij_reshape.repeat(L_bucket_row,1,1,1)
        mask_rij_sample = point_sample(rij_reshape.float(),rp_buckets_mask.float(),mode='bilinear')
        # 1, H*H_dim,L_bucket_row, L_bucket_column
        mask_rij_sample = mask_rij_sample.squeeze(0).reshape(r_h,r_hdim,L_bucket_row,L_bucket_column)
        mask_allrij_sample = mask_rij_sample.new_empty(r_h,r_hdim,L,L)
        mask_allrij_sample[:,:,1:,1:]=mask_rij_sample
        mask_allrij_sample[:,:,0,:] = rij_cls_token
        mask_allrij_sample[:,:,:,0] = rij_cls_token


        #similarly, sample the nonmask rij
        rp_buckets_nonmask = torch.clamp(rp_buckets_nonmask, min=-self.rpe_config.beta, max=self.rpe_config.beta)
        rp_buckets_nonmask = rp_buckets_nonmask/self.rpe_config.beta
        L_bucket_row, L_bucket_column, _ = rp_buckets_nonmask.shape
        rp_buckets_nonmask = rp_buckets_nonmask.unsqueeze(0)# 1, L_bucket_row, L_bucket_column, 2
        nonmask_rij_sample = point_sample(rij_reshape.float(),rp_buckets_nonmask.float(),mode='bilinear')
        nonmask_rij_sample = nonmask_rij_sample.squeeze(0).reshape(r_h,r_hdim,L_bucket_row,L_bucket_column)
        nonmask_allrij_sample = nonmask_rij_sample.new_empty(r_h,r_hdim,L,L)
        nonmask_allrij_sample[:,:,1:,1:]=nonmask_rij_sample
        nonmask_allrij_sample[:,:,0,:] = rij_cls_token
        nonmask_allrij_sample[:,:,:,0] = rij_cls_token

        x_input = x.permute(1,2,0,3) #H,L,B,head_dim
        #r_h,r_hdim,L,L
        mask_allrij_sample = mask_allrij_sample.permute(0,2,1,3)#H,L,head_dim,L_out


        #mask_rpe in shape H,L,B,L2
        mask_rpe = torch.matmul(
            x_input,
            mask_allrij_sample #H, H_dim, L
        )
        #get H,L,B,L_out
        mask_rpe = mask_rpe.permute(2,0,1,3)# B,H,L,L_out

        nonmask_allrij_sample = nonmask_allrij_sample.permute(0,2,1,3)
        nonmask_rpe = torch.matmul(
            x_input,
            nonmask_allrij_sample
        )
        nonmask_rpe = nonmask_rpe.permute(2,0,1,3)# B,H,L,L_out

            #.reshape(H,B,L,L_out).transpose(0, 1)
        #output: B,H,L,L_out
        #nonmask_rpe = nonmask_rpe.permute(2,0,1,3)# B,H,L,L2
        #mask input B*L
        tmp_mask = mask.unsqueeze(1).repeat(1,H,1).unsqueeze(-1).repeat(1,1,1,L)
        return tmp_mask*mask_rpe+(1-tmp_mask)*nonmask_rpe


    def __repr__(self):
        return 'iRPE(head_dim={rpe.head_dim}, num_heads={rpe.num_heads}, \
mode="{rpe.mode}", method={rpe.method}, transposed={rpe.transposed}, \
num_buckets={rpe.num_buckets}, initializer={rpe.initializer}, \
rpe_config={rpe.rpe_config})'.format(rpe=self)


class iRPE_Cross(nn.Module):
    """The implementation of image relative position encoding (specific for Cross method).

    Parameters
    ----------
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    transposed: bool
        Whether to transpose the input feature.
        For iRPE on queries or keys, transposed should be `True`.
        For iRPE on values, transposed should be `False`.
    num_buckets: int
        The number of buckets, which store encodings.
    initializer: None or an inplace function
        [Optional] The initializer to `lookup_table`.
        Initalize `lookup_table` as zero by default.
    rpe_config: RPEConfig
        The config generated by the function `get_single_rpe_config`.
    """

    def __init__(self, method, **kwargs):
        super().__init__()
        assert method == METHOD.CROSS
        self.rp_rows = iRPE(**kwargs, method=METHOD.CROSS_ROWS)
        self.rp_cols = iRPE(**kwargs, method=METHOD.CROSS_COLS)

    def forward(self, x, height=None, width=None):
        """forward function for iRPE.
        Compute encoding on horizontal and vertical directions separately,
        then summarize them.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor whose shape is (B, H, L, head_dim),
            where B is batch size,
                  H is the number of heads,
                  L is the sequence length,
                    equal to height * width (+1 if class token exists)
                  head_dim is the dimension of each head
        height: int or None
            [Optional] The height of the input
            If not defined, height = floor(sqrt(L))
        width: int or None
            [Optional] The width of the input
            If not defined, width = floor(sqrt(L))

        Returns
        -------
        rpe_encoding: torch.Tensor
            Image Relative Position Encoding,
            whose shape is (B, H, L, L)
        """

        rows = self.rp_rows(x, height=height, width=width)
        cols = self.rp_cols(x, height=height, width=width)
        return rows + cols

    def __repr__(self):
        return 'iRPE_Cross(head_dim={rpe.head_dim}, \
num_heads={rpe.num_heads}, mode="{rpe.mode}", method={rpe.method}, \
transposed={rpe.transposed}, num_buckets={rpe.num_buckets}, \
initializer={rpe.initializer}, \
rpe_config={rpe.rpe_config})'.format(rpe=self.rp_rows)


def get_single_rpe_config(ratio=1.9,
                          method=METHOD.PRODUCT,
                          mode='contextual',
                          shared_head=True,
                          skip=0):
    """Get the config of single relative position encoding

    Parameters
    ----------
    ratio: float
        The ratio to control the number of buckets.
    method: METHOD
        The method ID of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    shared_head: bool
        Whether to share weight among different heads.
    skip: int [0 or 1]
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.

    Returns
    -------
    config: RPEConfig
        The config of single relative position encoding.
    """
    config = edict()
    # whether to share encodings across different heads
    config.shared_head = shared_head
    # mode: None, bias, contextual
    config.mode = mode
    # method: None, Bias, Quant, Cross, Product
    config.method = method
    # the coefficients of piecewise index function
    config.alpha = 1 * ratio
    config.beta = 2 * ratio
    config.gamma = 8 * ratio

    # set the number of buckets
    config.num_buckets = get_num_buckets(method,
                                         config.alpha,
                                         config.beta,
                                         config.gamma)
    # add extra bucket for `skip` token (e.g. class token)
    config.num_buckets += skip
    return config


def get_rpe_config(ratio=1.9,
                   method=METHOD.PRODUCT,
                   mode='contextual',
                   shared_head=True,
                   skip=0,
                   rpe_on='k'):
    """Get the config of relative position encoding on queries, keys and values

    Parameters
    ----------
    ratio: float
        The ratio to control the number of buckets.
    method: METHOD or str
        The method ID (or name) of image relative position encoding.
        The `METHOD` class is defined in `irpe.py`.
    mode: str or None
        The mode of image relative position encoding.
        Choices: [None, 'bias', 'contextual']
    shared_head: bool
        Whether to share weight among different heads.
    skip: int [0 or 1]
        The number of skip token before spatial tokens.
        When skip is 0, no classification token.
        When skip is 1, there is a classification token before spatial tokens.
    rpe_on: str
        Where RPE attaches.
        "q": RPE on queries
        "k": RPE on keys
        "v": RPE on values
        "qk": RPE on queries and keys
        "qkv": RPE on queries, keys and values

    Returns
    -------
    config: RPEConfigs
        config.rpe_q: the config of relative position encoding on queries
        config.rpe_k: the config of relative position encoding on keys
        config.rpe_v: the config of relative position encoding on values
    """

    # alias
    if isinstance(method, str):
        method_mapping = dict(
            euc=METHOD.EUCLIDEAN,
            quant=METHOD.QUANT,
            cross=METHOD.CROSS,
            product=METHOD.PRODUCT,
        )
        method = method_mapping[method.lower()]
    if mode == 'ctx':
        mode = 'contextual'
    config = edict()
    # relative position encoding on queries, keys and values
    kwargs = dict(
        ratio=ratio,
        method=method,
        mode=mode,
        shared_head=shared_head,
        skip=skip,
    )
    config.rpe_q = get_single_rpe_config(**kwargs) if 'q' in rpe_on else None
    config.rpe_k = get_single_rpe_config(**kwargs) if 'k' in rpe_on else None
    config.rpe_v = get_single_rpe_config(**kwargs) if 'v' in rpe_on else None
    return config


def build_rpe(config, head_dim, num_heads):
    """Build iRPE modules on queries, keys and values.

    Parameters
    ----------
    config: RPEConfigs
        config.rpe_q: the config of relative position encoding on queries
        config.rpe_k: the config of relative position encoding on keys
        config.rpe_v: the config of relative position encoding on values
        None when RPE is not used.
    head_dim: int
        The dimension for each head.
    num_heads: int
        The number of parallel attention heads.

    Returns
    -------
    modules: a list of nn.Module
        The iRPE Modules on [queries, keys, values].
        None when RPE is not used.
    """
    if config is None:
        return None, None, None
    rpes = [config.rpe_q, config.rpe_k, config.rpe_v]
    transposeds = [True, True, False]

    def _build_single_rpe(rpe, transposed):
        if rpe is None:
            return None

        rpe_cls = iRPE if rpe.method != METHOD.CROSS else iRPE_Cross
        return rpe_cls(
            head_dim=head_dim,
            num_heads=1 if rpe.shared_head else num_heads,
            mode=rpe.mode,
            method=rpe.method,
            transposed=transposed,
            num_buckets=rpe.num_buckets,
            rpe_config=rpe,
        )
    return [_build_single_rpe(rpe, transposed)
            for rpe, transposed in zip(rpes, transposeds)]


if __name__ == '__main__':
    config = get_rpe_config(skip=1)
    rpe = build_rpe(config, head_dim=32, num_heads=4)
    print(rpe)
