from modules.Fourier_modules import *
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

