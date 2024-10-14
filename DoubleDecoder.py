import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,esDecoderLayer,RNNWithInit
from layers.SelfAttention_Family import ProbAttention, AttentionLayer,MAM
from layers.Embed import DataEmbedding
# from efficient_kan import kan
import pywt
import config
def Dwt_for_Signals(x,wavename):
    x =x.permute(0,2,1)
    B,C,T = x.size()
    wavename = "db3"
    CA = torch.empty(B,1,28)
    CD = torch.empty(B,1,52)
    x = x.cpu().detach().numpy()
    for i in range(C):
        cA,cD = pywt.dwt(x[:,i,:],wavename,axis=-1)
        [cA1,cD1,_] = pywt.wavedec(x[:,i,:],wavename,level=2,axis=-1)
        cD1 = torch.tensor(cD1).unsqueeze(1);cD = torch.tensor(cD).unsqueeze(1)
        CA = torch.concatenate((CA,cD1),dim=1)
        CD = torch.concatenate((CD,cD),dim=1)
    wt = torch.concat((CA[:,1:,:],CD[:,1:,:]),dim=2)
    return wt 
class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """
    def __init__(self,configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.dec_one_out =configs.es_out
        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        #encoder 
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        #decoder layer 1 default only 1 layers has configs.d_model 
        # outputis pose with noisy
        self.posedecoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            # projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            projection= nn.Linear(configs.d_model,configs.c_out)
        )
        # decoder layer 2  
        # output is pred
        self.esDecoder = esDecoderLayer(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            # projection=nn.Linear(configs.seq_len*configs.enc_in, configs.es_out, bias=True) 
            
            projection=nn.Linear(configs.seq_len*configs.enc_in, configs.es_out)
        )
        
        self.device = configs.device
        
        self.rnnInit = RNNWithInit(input_size=configs.es_out, output_size=configs.c_out,hidden_size=configs.rnn_hidden ,num_rnn_layer=1, rnn_type='LSTM', bidirectional=True, dropout=configs.dropout_rnn)
    def long_forecast(self,x_enc,x_mark_enc,x_dec,x_mark_dec):
        # TFD_out = Dwt_for_Signals(x_enc,"symmetric")
        # # TFD_out = self.batchnorm(TFD_out).permute(0,2,1)
        # # TFD_out = self.MaM(TFD_out)
        # TFD_out = TFD_out.to(self.device)
        # zeros_TFD = torch.zeros_like(x_enc)
        # new_TFD = torch.concat([zeros_TFD.permute(0,2,1)[:,:,:20],TFD_out],dim=2)
        # x_encs = torch.concat([new_TFD.permute(0,2,1),x_enc],dim=2)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out,attens = self.encoder(enc_out, attn_mask=None)
        pose_noisy = self.posedecoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        pose_es = self.esDecoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        HandPose = self.rnnInit(pose_noisy,pose_es)
        return HandPose,pose_es
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,mask=None):
        dec_out,pose_es = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out,pose_es
