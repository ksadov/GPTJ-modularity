from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJAttention
import torch
from input_texts import articles, papers, code
from typing import List
from plotting import stream_plot, get_dots_from_tokens, plot_dots

def setup_model(device, repo, use_fp16):
    print("Getting model")
    if use_fp16:
        model = AutoModel.from_pretrained(repo, torch_dtype=torch.float16)
    else:
        model = AutoModel.from_pretrained(repo)
    print("moving model to {}".format(device))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    predicter = AutoModelForCausalLM.from_pretrained(repo)
    return model, tokenizer, predicter

def register_activation_hooks(model):
    activation = {}

    def detached( output ):
      if type(output) is tuple:
        return ( detached(out) for out in output )
      
      if type(output) is torch.Tensor:
        return output.detach()
      
      return None

    def get_activation( name ):
      def hook(model, input, output):
        if not type( output ) is tuple:
          return     
        output = detached( output )
        activation[name] = output
      return hook

    def pad_zeros( d, n=2 ):
      s = str(d)
      k = n - len(s)
      k = k if k > 0 else 0
      return "0"*k + s

    # register the forward hook
    decoder_index   = 0
    attention_index = 0
    for module in model.h: 
      module = module.attn
      if type(module) is GPTJAttention:
        name = pad_zeros( attention_index ) + "-attention" 
        module.register_forward_hook( get_activation( name ) )
        attention_index += 1
        continue

    return activation
  
def get_inputs_embeds( text, model, tokenizer, verbose=False, limit=None ):
  inputs = tokenizer(text, return_tensors="pt")

  input_ids = inputs.input_ids
  
  if verbose >= 2:
    print("inputs:")
    print( inputs.input_ids.size() )
  
  if limit:
    prev_size = input_ids.size()
    input_ids = input_ids[0][:limit].reshape(1, -1)
    new_size = input_ids.size()

    if verbose == 1:
      print("trimmed from", list(prev_size), "to", list(new_size) )

    if verbose >= 2:
      print("trimmed inputs:")
      print( new_size )

  inputs_embeds = model.wte(input_ids.to(model.device))
  
  if verbose >= 2:
    print( inputs_embeds.size() )

  return inputs_embeds

def predict( text, tokenizer, predicter, num=10, limit=None ):
  inputs = tokenizer( text, return_tensors="pt" )
  input_ids = inputs.input_ids

  if limit:
    input_ids = input_ids[0][:limit].reshape(1, -1)
  
  generate_ids = predicter.generate( input_ids, max_length=len(input_ids[0])+num )
  
  before = tokenizer.batch_decode( input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  after  = tokenizer.batch_decode( generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  after = after[len(before):]
  return before, after

def get_recent_activations(activation):
  layers = []
  for key, value in activation.items():
    layer = []
    layer.append( key )
    #print( key, value, type(value) )
    for out in value:
      if type(out) is torch.Tensor:
        layer.append( out )
        continue
  
      if out is None:
        continue
  
      for o in out:
        layer.append( o )
  
    layers.append(layer)
  return layers

def get_text_activations( text, model, tokenizer, activation, verbose=False, limit=None, **kwargs ):
  inputs_embeds = get_inputs_embeds( text, model, tokenizer, verbose, limit )

  # run the model
  outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True, **kwargs )

  # get the hidden states
  hidden_states = torch.stack( outputs.hidden_states )
  hidden_states = hidden_states.reshape([ -1, *hidden_states.size()[-2:] ]).detach()
  input = hidden_states[0]

  # get attention outputs
  attention_out = torch.stack([ out[1] for out in get_recent_activations(activation) ])
  attention_out = attention_out.reshape([-1, *attention_out.size()[-2:] ]).detach()

  # get ff outputs
  ff_out =  [] 
  for i in range(len(attention_out)):
    ff_out.append( hidden_states[i+1] - attention_out[i] - hidden_states[i] )
  ff_out = torch.stack( ff_out )
  ff_out = ff_out.reshape([-1, *ff_out.size()[-2:] ])

  output = outputs.last_hidden_state[0].detach()

  return input, attention_out, ff_out, output

def get_residual_stream( text, model, tokenizer, activation, verbose=False, limit=None, **kwargs ):
  inputs_embeds = get_inputs_embeds( text, model, tokenizer, verbose, limit )

  # run the model
  outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True, **kwargs )

  # get the hidden states
  hidden_states = torch.stack( outputs.hidden_states )
  hidden_states = hidden_states.reshape([ -1, *hidden_states.size()[-2:] ]).detach()
  input = hidden_states[0]

  # get attention outputs
  attention_out = torch.stack([ out[1] for out in get_recent_activations(activation) ])
  attention_out = attention_out.reshape([-1, *attention_out.size()[-2:] ]).detach()

  # build residual stream
  residual_stream = []
  residual_stream.append( input )
  for i in range(len( attention_out )):
    residual_stream.append( (residual_stream[-1] + attention_out[i]) )
    residual_stream.append( (hidden_states[i+1]) )

  return torch.stack( residual_stream )

def get_tokens_from_layer_from_outputs(
    hidden_states,
    layer_index: int,
    output_index: int=1,
    sub_indices: List[int]=[-1],
    head: int=0 ):
  num_texts = len( hidden_states )

  size = hidden_states[0][layer_index][output_index][0].size()
  # find number of attention heads
  if len( size ) == 3:
    num_heads = size[0]
  else:
    num_heads = 1

  # define shorthand for getting the value
  def out( text_index, head=0, sub_index=-1 ):
    if num_heads == 1:
      return hidden_states[text_index][layer_index][output_index][0][sub_index].flatten()
    
    return hidden_states[text_index][layer_index][output_index][0][head][sub_index].flatten()

  tokens = []

  for i in range( num_texts ):
    for p in sub_indices:
        a = out(i, head, p)
        tokens.append( a )
  
  return tokens

def get_tokens(
    hidden_states,
    layer_index: int,
    sub_indices: List[int]=[-1],
    head: int=0 ):
  num_texts = len( hidden_states )

  size = hidden_states[0][layer_index].size()
  # find number of attention heads
  if len( size ) == 2:
    num_heads = size[0]
  else:
    num_heads = 1

  # define shorthand for getting the value
  def out( text_index, head=0, sub_index=-1 ):
    if num_heads == 1:
      return hidden_states[text_index][layer_index][sub_index].flatten()
    
    return hidden_states[text_index][layer_index][head][sub_index].flatten()

  tokens = []

  for i in range( num_texts ):
    for p in sub_indices:
        a = out(i, head, p)
        tokens.append( a )
  
  return tokens

def print_predictions(articles, papers, code, tokenizer, predicter):
  """
  Debug function to check that our model is performing inference correctly
  """
  sub_indices = list(range(-1-9*4, 0, 4))
  for text in [ *articles, *papers, *code ]:
    print( "########### now printing new thing:" )
    for i, x in enumerate(sub_indices):
      limit = 500+x
      print( i, f": '{predict( text, tokenizer, predicter, 10, limit )[1]}'" )


def make_plots(articles, papers, code, model, tokenizer, predicter, activation):
  sub_indices = list(range(-1-(9*32), 0, 32))
  residual_states = []
  for post in [ *articles, *papers, *code ]:
      residual_states.append( get_residual_stream( post, model, tokenizer, activation, verbose=0, limit=512 ) )
  residual_states = torch.stack( residual_states )

  # the original post checks dimensions 13, 69, 256, 0-4
  # for now we'll just check 0-4
  for dimension in [*list(range(5)) ]:
      fname = f"dimension_{dimension}.png"
      stream_plot( fname, residual_states, dimension, sub_indices )
  lists = [ [], [], [], [] ]

  for text in [ *articles, *papers, *code ]:
    activations = [input, attn, ff, out] = get_text_activations( text, model, tokenizer, activation, verbose=0, limit=512 )
    for i, act in enumerate(activations):
      lists[i].append( act )

  for i in range(len(lists)):
    lists[i] = torch.stack( lists[i] )

  [ all_inputs, all_attn_out, all_ff_out, all_outputs ] = lists

  num_texts = len(all_attn_out)
  num_layers = len(all_attn_out[0])
  dots_0 = []
  dots_1 = []
  for layer in range(num_layers):
    outputs = []
    for text in range(num_texts):
      for sub_index in sub_indices:
        outputs.append( all_attn_out[text][layer][sub_index].detach() )
    dots_0.append( get_dots_from_tokens( outputs, "cosine" ) )
    dots_1.append( get_dots_from_tokens( outputs, "scaled" ) )

  fname0 = "attention-cosine-similarity.png"
  fname1 = "attention-scaled-similarity.png"

  print(f"Saving cosine similarity between activations to {fname0}")
  plot_dots( dots_0, 4, tick_skip=10, save_name=fname0)

  print(f"Saving scaled similarity between activations to {fname1}")
  plot_dots( dots_1, 4, tick_skip=10, save_name=fname1)

def main(): 
    #repo = "EleutherAI/gpt-j-6B"
    # 6B is giving me OOM errors, so I use this small model for testing
    repo = "hf-internal-testing/tiny-random-gptj"
    use_fp16 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, predicter = setup_model(device, repo, use_fp16)
    activation = register_activation_hooks(model)
    #print_predictions(articles, papers, code, tokenizer, predicter)
    make_plots(articles, papers, code, model, tokenizer, predicter, activation)

if __name__ == "__main__":
    main()