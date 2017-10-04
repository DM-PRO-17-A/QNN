import numpy as np

# adapted from A. Karpathy's CS231 im2col code
# utilities to generate a patch matrix from a multichannel image
# of shape (batches, channels, height, width)

def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride_y=1, stride_x=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  print(N,C,H,W)
  # 1,3,32,32
  assert (H + 2 * padding - field_height) % stride_y == 0
  assert (W + 2 * padding - field_width) % stride_x == 0
  out_height = (H + 2 * padding - field_height) / stride_y + 1
  #14
  out_width = (W + 2 * padding - field_width) / stride_x + 1
  #14
  i0 = np.repeat(np.arange(field_height), field_width)
  #[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
  i0 = np.tile(i0, C)
  #repeats the vektor 3 times
  #[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
  i1 = stride_y * np.repeat(np.arange(out_height), out_width)
  #28*[0-27]
  j0 = np.tile(np.arange(field_width), field_height * C)
  #[0 1 2 3 4] <- 15 times
  j1 = stride_x * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width)
  k = k.reshape(-1, 1)
  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=0, stride_y=1, stride_x=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  #padding = 0
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
  #xpadded = x
  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride_y, stride_x)
  #k = 25*[0,1,2] dim: (75,1)
  #i = 14[0,2,4....27] dim (75,784)
  #j = [[0-27][1-28]....[4-31]] dim: (75,784)
  cols = x_padded[:, k, i, j]
  #cols = masse tall rundt paa 70-200, dim (1,75,784) vet ikke hva den gjor
  C = x.shape[1]
  #C=3
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  #cols dim: (75,784)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=0,
                   stride_y=1, stride_x=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride_y, stride_x)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]
