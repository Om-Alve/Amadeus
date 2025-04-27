 from transformers import PretrainedConfig

 class AmadeusConfig(PretrainedConfig):
     model_type = "amadeus"

     def __init__(
         self,
         dropout: float=0.2,
         bos_token_id: int=0,
         eos_token_id: int=1,
         hidden_act: str="silu",
         hidden_size: int=512,
         intermediate_size: int=None,
         max_position_embeddings: int=32768,
         num_attention_heads: int=8,
         num_hidden_layers: int=12,
         num_key_value_heads: int=2,
         vocab_size: int = 6400,
         rms_norm_eps: float = 1e-05,
         rope_theta: int = 1000000.0,
         flash_attn: bool = True,
     ):
         super().__init__()
         self.dropout = dropout
         self.bos_token_id = bos_token_id
         self.eos_token_id = eos_token_id
         self.hidden_act = hidden_act
         self.hidden_size = hidden_size
         self.intermediate_size = intermediate_size if intermediate_size is not None else hidden_size * 4
         self.max_position_embeddings = max_position_embeddings
         self.num_attention_heads = num_attention_heads
         self.num_hidden_layers = num_hidden_layers
         self.num_key_value_heads = num_key_value_heads
         self.vocab_size = vocab_size
         self.rms_norm_eps = rms_norm_eps
         self.rope_theta = rope_theta
         self.flash_attn = flash_attn

     
