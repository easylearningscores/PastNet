import torch
from utils import *
import logging
from torch import nn
from modules.DiscreteSTModel_modules import *
from modules.Fourier_modules import *



def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker,
                          stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

class FPG(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=20,
                 out_channels=20,
                 input_frames=20,
                 embed_dim=768,
                 depth=12,
                 mlp_ratio=4.,
                 uniform_drop=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 dropcls=0.):
        super(FPG, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = input_frames
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_c=in_channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # [1, 196, 768]
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = self.patch_embed.grid_size[0]
        self.w = self.patch_embed.grid_size[1]
        '''
        stochastic depth decay rule
        '''
        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h=self.h,
            w=self.w)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.linearprojection = nn.Sequential(OrderedDict([
            ('transposeconv1', nn.ConvTranspose2d(embed_dim, out_channels * 16, kernel_size=(2, 2), stride=(2, 2))),
            ('act1', nn.Tanh()),
            ('transposeconv2', nn.ConvTranspose2d(out_channels * 16, out_channels * 4, kernel_size=(2, 2), stride=(2, 2))),
            ('act2', nn.Tanh()),
            ('transposeconv3', nn.ConvTranspose2d(out_channels * 4, out_channels, kernel_size=(4, 4), stride=(4, 4)))
        ]))

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        '''
        patch_embed:
        [B, T, C, H, W] -> [B*T, num_patches, embed_dim]
        '''
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.patch_embed(x)
        #enc = LearnableFourierPositionalEncoding(768, 768, 64, 768, 10)
        #fourierpos_embed = enc(x)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.pos_drop(x + fourierpos_embed)


        if not get_fourcastnet_args().checkpoint_activations:
            for blk in self.blocks:
                x = blk(x)
        else:
            x = checkpoint_sequential(self.blocks, 4, x)

        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, self.h, self.w])
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.linearprojection(x)
        x = x.reshape(B, T, C, H, W)
        return x

class DST(nn.Module):
    def __init__(self,
                 in_channel=1,
                 num_hiddens=128,
                 res_layers=2,
                 res_units=32,
                 embedding_nums=512,  # K
                 embedding_dim=64,  # D
                 commitment_cost=0.25):
        super(DST, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = embedding_nums
        self._encoder = Encoder(in_channel, num_hiddens,
                                res_layers, res_units)  #
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        # code book
        self._vq_vae = VectorQuantizerEMA(embedding_nums,
                                          embedding_dim,
                                          commitment_cost,
                                          decay=0.99)

        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                res_layers,
                                res_units,
                                in_channel)

    def forward(self, x):
        # input shape : [B, C, W, H]
        z = self._encoder(x)  # [B, hidden_units, W//4, H//4]
        # [B, embedding_dims, W//4, H//4] z -> encoding
        z = self._pre_vq_conv(z)
        # quantized -> embedding, quantized相当于videoGPT中的 encoder输出
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

    def get_embedding(self, x):
        return self._pre_vq_conv(self._encoder(x))

    def get_quantization(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        _, quantized, _, _ = self._vq_vae(z)
        return quantized

    def reconstruct_img_by_embedding(self, embedding):
        loss, quantized, perplexity, _ = self._vq_vae(embedding)
        return self._decoder(quantized)

    def reconstruct_img(self, q):
        return self._decoder(q)

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder


class DynamicPropagation(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(DynamicPropagation, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(
            channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(
                channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid //
                          2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(
            channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(
                2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid //
                          2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, input_state):
        B, T, C, H, W = input_state.shape
        input_state = input_state.reshape(B, T*C, H, W)
        # encoder
        skips = []
        hidden_embed = input_state
        for i in range(self.N_T):
            hidden_embed = self.enc[i](hidden_embed)
            if i < self.N_T - 1:
                skips.append(hidden_embed)

        # decoder
        hidden_embed = self.dec[0](hidden_embed)
        for i in range(1, self.N_T):
            hidden_embed = self.dec[i](torch.cat([hidden_embed, skips[-i]], dim=1))

        output_state = hidden_embed.reshape(B, T, C, H, W)
        return output_state


class PastNetModel(nn.Module):
    def __init__(self, 
                 args,
                 shape_in, 
                 hid_T=256, 
                 N_T=8, 
                 incep_ker=[3, 5, 7, 11], 
                 groups=8, 
                 res_units=64, 
                 res_layers=2, 
                 embedding_nums=512, 
                 embedding_dim=64):
        super(PastNetModel, self).__init__()
        T, C, H, W = shape_in
        self.DST_module = DST(in_channel=C,
                             res_units=res_units,
                             res_layers=res_layers,
                             embedding_dim=embedding_dim,
                             embedding_nums=embedding_nums)

        self.FPG_module = FPG(img_size=64,
                              patch_size=16,
                              in_channels=1,
                              out_channels=1,
                              embed_dim=128,
                              input_frames=10,
                              depth=1,
                              mlp_ratio=2.,
                              uniform_drop=False,
                              drop_rate=0.,
                              drop_path_rate=0.,
                              norm_layer=None,
                              dropcls=0.)

        if args.load_pred_train:
            print_log("Load Pre-trained Model.")
            self.vq_vae.load_state_dict(torch.load("./models/vqvae.ckpt"), strict=False)

        if args.freeze_vqvae:
            print_log(f"Params of VQVAE is freezed.")
            for p in self.vq_vae.parameters():
                p.requires_grad = False
        self.DynamicPro = DynamicPropagation(T*64, hid_T, N_T, incep_ker, groups)

    def forward(self, input_frames):
        B, T, C, H, W = input_frames.shape
        pde_features = self.FPG_module(input_frames)
        input_features = input_frames.view([B * T, C, H, W])
        encoder_embed = self.DST_module._encoder(input_features)
        z = self.DST_module._pre_vq_conv(encoder_embed)
        vq_loss, Latent_embed, _, _ = self.DST_module._vq_vae(z)

        _, C_, H_, W_ = Latent_embed.shape
        Latent_embed = Latent_embed.reshape(B, T, C_, H_, W_)

        hidden_dim = self.DynamicPro(Latent_embed)
        B_, T_, C_, H_, W_ = hidden_dim.shape
        hid = hidden_dim.reshape([B_ * T_, C_, H_, W_])

        predicti_feature = self.DST_module._decoder(hid)
        predicti_feature = predicti_feature.reshape([B, T, C, H, W]) + pde_features

        return predicti_feature


