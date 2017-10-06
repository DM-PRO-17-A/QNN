import numpy as np

# adapted from A. Karpathy's CS231 im2col code
# utilities to generate a patch matrix from a multichannel image
# of shape (batches, channels, height, width)

def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride_y=1, stride_x=1):
  # First figure out what the size of the output should be
  
  # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx
  #  Alle kommentarene bruker tall fra 1 gang den blir kjort
  # XXXXXXXXXXXXXXXXXXXXXXXx

  N, C, H, W = x_shape
  # 1,3,32,32
  assert (H + 2 * padding - field_height) % stride_y == 0
  assert (W + 2 * padding - field_width) % stride_x == 0
  # Ser om det lille vinduet faar plass i  det store vinduet
  out_height = (H + 2 * padding - field_height) / stride_y + 1
  #28 Hvor mange ganger faar det lille vinduet plass i det store i hoyden
  out_width = (W + 2 * padding - field_width) / stride_x + 1
  #28 Hvor mange ganger faar det lille vinduet plass i det store i bredden
  i0 = np.repeat(np.arange(field_height), field_width)
  #[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
  # (1,25) repeter bildehoyden med bildelengden. (5x5)
  i0 = np.tile(i0, C)
  #repeats the vektor 3 times
  #[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
  #(1,75), samme tall som rader paa det ferdige produktet
  
  i1 = stride_y * np.repeat(np.arange(out_height), out_width)
  # 1 * (784,)
  # (784,) = repeat [0-27] 28 times. [0 *28, 1*28, 2*28.....] 

  j0 = np.tile(np.arange(field_width), field_height * C)
  # repeat [0-4] 5*3 times
  #[0 1 2 3 4] <- 15 times
  #(75,)

  j1 = stride_x * np.tile(np.arange(out_width), out_height)
  # Tallene 0-27 repetert etterhverader 28 ganger. [0 1 2 3 4 5.....0 1 2 3 4...]
  #(784) 

  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  # omgjor (1,75) til (75,1) og plusser paa (784,1) slik at det blir en
  # matrise paa (75,784) som ser slik ut:
  # [[0-27]
  #  [0-27]
  #  .
  #  .
  #  [4-31]]
  
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  # (75,1) + (784,1) = (75,784) som ser slik ut:
  # [
  # [ 0 1 2 -> 27  repetert 28 ganger ]
  # [ 1 2 3 -> 28  repetert 28 ganger ]
  # [ 2 3 4 -> 29  repetert 28 ganger ]
  # [ 3 4 5 -> 30  repetert 28 ganger ]
  # [ 4 5 6 -> 31  repetert 28 ganger ] ........]
  # Disse er repetert 15 ganger og de oppgjor dermed 
  # (5*15 = 75, 28*28 = 784) 
  # 
  k = np.repeat(np.arange(C), field_height * field_width)
  # repeterer [0 1 2] 25 ganger, altsaa [0 0 0 0 0 ... 1 1 1 1 1.. 2 2..]
  # (1,75)
  k = k.reshape(-1, 1)
  # omgjor det flate vektoren til (75,1)

  return (k, i, j)
  #returnerer: 
  # (75,1) med [0][0]...[1][1]..[2][2]..
  # (75, 784) med [[0-27][0-27]...[1-28].....[4-31]]
  # (75, 784) med [[0-27*28][1-28*28]...[4-31*28] * 15]

def im2col_indices(x, field_height, field_width, padding=0, stride_y=1, stride_x=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  #padding = 0
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
  #xpadded = x
  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride_y, stride_x)
  #k = (75,1) med [0][0]...[1][1]..[2][2]..
  #i = (75, 784) med [[0-27][0-27]...[1-28].....[4-31]]
  #j = (75, 784) med [[0-27*28][1-28*28]...[4-31*28] * 15]

  cols = x_padded[:, k, i, j]

  # x_padded dim: (1,3,32,32)
  # cols dim: (1,75,784)
  # Inneholder verdiene fra det originale bilde som har blitt plassert
  # paa indeksene 

  # Bruker multiple-dim-arrayer som index. Det er aa bruke taallene inne i index-
  # arrayene og hente ut de tallene fra arrayen. 
  # x_padded[:] = (1,3,32,32)
  # x_padded[:,k] = (1,75,1,32,32)
  # x_padded[:,k,i] = (1, 75, 784, 32)
  # x_padded[:,k,i,j] = (1, 75, 784)

  # Den lager en ny matrise med 75 rader med tall fra 
  
  C = x.shape[1]
  #C=3
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  
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
