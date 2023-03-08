import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
import torch
import copy

norm = lambda x : torch.linalg.norm( x )

def normalize(v, _norm=None ):
  v = copy.copy( v )
  if _norm is None:
    _norm = norm( v )
  if _norm == 0:
      _norm = np.finfo(v.dtype).eps
  return v/_norm

def get_dots_from_tokens( tokens, similarity_metric="cosine" ):
  num_tokens = len(tokens)
  dots = np.zeros( (num_tokens, num_tokens) )

  if similarity_metric == "scaled":
    norms = [ norm( t.to('cpu') ) for t in tokens ]
    max_norm = np.max( norms )
    normalized_tokens = [ normalize( t, max_norm ) for t in tokens ]

  if similarity_metric == "cosine":
    normalized_tokens = [ normalize(t) for t in tokens ]
  
  for i, t_i in enumerate(normalized_tokens):
    for j, t_j in enumerate(normalized_tokens):
      dots[i][j] = torch.dot( t_i, t_j )
  
  return dots

def get_dots( hidden_states, layer_index=0, output_index=1, sub_indices=[-1], similarity_metric="cosine" ):
  # get the actual dot products
  # cosine = all vectors normalized to themselves
  # scaled = all vectors normalized to the largest vector
  num_texts = len( hidden_states )

  size = outputs[0][ layer_index ][ output_index ][0].size()
  # find number of attention heads
  if len( size ) == 3:
    num_heads = size[0]
  else:
    num_heads = 1

  dots = []
  for n in range( num_heads ):
    tokens = get_tokens_from_layer_from_outputs( hidden_states, layer_index, output_index, sub_indices, head=n )
    dots.append( get_dots_from_tokens(tokens, similarity_metric) )

  return dots

def get_all_dots( hidden_states,
                  output_index: int = 1,
                  sub_indices: List[int] = [-1],
                  similarity_metric: str = "cosine",
                  plot=False, save=False, name=None ):
  
  # handle error case
  if len(hidden_states) == 0:
    print("no hidden given...")
    return
  
  #get some labels and stuff
  NUM_LAYERS = len( hidden_states[0] )
  output_index_map = {1: "final-output", 2: "self-attention-weights", 3: "value-key"}
  output_label = output_index_map[output_index]

  dots = []

  for layer_index in range( NUM_LAYERS ):
    # get name
    if name is None:
      name = hidden_states[0][layer_index][0]

    # get dot products
    layer_dots = get_dots( hidden_states, layer_index, output_index, sub_indices, similarity_metric )
    dots.append( layer_dots )

    # plot if needed
    if plot:
      fig = plot_dots( dots )
      if save:
        fig.savefig( PATH + f"layer{name}-{output_label}-dots-{similarity_metric}.png" )

  return dots

def plot_dots( dots, max_per_row=16, tick_skip=None, plot_mode="heatmap", cmap=None, save_name=None ):
  num_heads = len(dots)
  # find number of subplots
  if num_heads <= max_per_row:
    num_rows = 1
    num_cols = num_heads
  else:
    num_rows = num_heads // max_per_row
    num_cols = max_per_row

  # create subplots
  figsize = ( 4*num_cols + 2, 4*num_rows )
  fig, ax = plt.subplots( num_rows, num_cols, figsize=figsize )

  # function to get subplot index
  def plot_index( head_number ):
    row = 0
    curr = head_number
    while curr >= num_cols:
      curr -= num_cols
      row += 1

    return row, curr

  for n in range( num_heads ):
    x, y = plot_index( n )
    curr_ax = ax
    if num_rows > 1:
      curr_ax = curr_ax[x]
    if num_cols > 1:
      curr_ax = curr_ax[y]

    LN = len(dots[n][0])
    x_ticks = list(range( LN           ))
    y_ticks = list(range( len(dots[n]) ))

    if plot_mode == "heatmap":
      sns.heatmap( dots[n], vmin=-1, center=0, vmax=1, ax=curr_ax,
                  xticklabels=x_ticks, yticklabels=y_ticks )
    if plot_mode == "norm":
      curr_ax.plot([ dots[n][i][i] for i in range(len(dots[n])) ], scalex=True, scaley=True)

    # handle plot ticks
    if tick_skip:
      xticks, yticks = curr_ax.xaxis.get_major_ticks(), curr_ax.yaxis.get_major_ticks()
      for t in [ xticks, yticks ]:
        for i in range(len(t)):
          if i%tick_skip == 0 or i == LN-1:
            continue
          t[i].set_visible(False)

  if save_name is not None:
    plt.savefig( save_name, dpi=300 )

  return fig


def scatter_plot( hidden_states, sub_indices=[-1], skips=None ):
  for layer in range( 1, len(hidden_states[0]) ):
  #for layer in [ 1 ]:
    tokens = get_tokens_from_layer_from_outputs( hidden_states, layer, 1, sub_indices )
    tokens = torch.tensor([ [*t][:100] for t in tokens ]) 
    # [ 30, 768 ] -> [ 768, 30 ]

    items = []
    if skips is None:
      skips = set( range(0, len(tokens), len(sub_indices) ) ) 

    for i in range(len(tokens)):
      if i in skips:
        items.append([])
      items[-1].append( list( enumerate( tokens[i] )  ) )
    
    plt.figure(figsize=(30, 15))

    colors = ["r", "g", "b"]

    for i in range(len(items)):
      items[i] = torch.tensor( items[i] )
      points = items[i]
      points = points.reshape( (-1, 2) )
      [x, y ] = points.permute( 1, 0 )

      plt.scatter( x, y, s=4, c=colors[i] )

  
def stream_plot( fname, hidden_states, dimension, sub_indices=[-1], skips=None, figsize=(8,15), ylim=None ):
  NUM_TEXTS = len(hidden_states)
  NUM_LAYERS = len(hidden_states[0])
  NUM_SUBINDICES = len(sub_indices)
  
  cmap = [ "firebrick", "red", "orangered", "indianred", "darkgreen", "green", "limegreen", "midnightblue", "darkslateblue", "royalblue" ]
  
  output_streams = []
  for text in range( NUM_TEXTS ):
    output_streams.append([])
    for sub_index in sub_indices:
      output_streams[-1].append([])
      for layer in range( NUM_LAYERS ):
        val = hidden_states[text][layer][sub_index][dimension]
        output_streams[-1][-1].append([ layer/2, val ])
  
  output_streams = torch.tensor( output_streams ).transpose( -1,-2 )
  
  plt.figure(figsize=(8, 15))

  for sub in range(NUM_SUBINDICES):
    for text in range(NUM_TEXTS):
      [ x, y ] = output_streams[text][sub]
      label = f"text {text}" if sub == 0 else None      
      plt.scatter( x, y, s=100, c=cmap[text], label=label, marker="_" )

  plt.xlabel("Layer")
  plt.ylabel(f"Value of Token Embedding Dimension {dimension}")
  plt.legend()
  plt.ylim( ylim )
  plt.savefig(fname)
  print(f"Saved stream plot to {fname}")