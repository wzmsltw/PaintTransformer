import paddle
import paddle.nn as nn
import math

class Painter(nn.Layer):
    """
    network architecture written in paddle.
    """
    def __init__(self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3):
        super().__init__()
        self.enc_img = nn.Sequential(
            nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(3, 32, 3, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(),  # maybe replace with the inplace version
            nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(32, 64, 3, 2),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(64, 128, 3, 2),
            nn.BatchNorm2D(128),
            nn.ReLU())
        self.enc_canvas = nn.Sequential(
            nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(3, 32, 3, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(32, 64, 3, 2),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Pad2D([1, 1, 1, 1], 'reflect'),
            nn.Conv2D(64, 128, 3, 2),
            nn.BatchNorm2D(128),
            nn.ReLU())
        self.conv = nn.Conv2D(128 * 2, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers)
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_per_stroke))
        self.linear_decider = nn.Linear(hidden_dim, 1)
        self.query_pos = paddle.static.create_parameter([total_strokes, hidden_dim], dtype='float32', 
                                                        default_initializer=nn.initializer.Uniform(0, 1))
        self.row_embed = paddle.static.create_parameter([8, hidden_dim // 2], dtype='float32', 
                                                        default_initializer=nn.initializer.Uniform(0, 1))
        self.col_embed = paddle.static.create_parameter([8, hidden_dim // 2], dtype='float32', 
                                                        default_initializer=nn.initializer.Uniform(0, 1))

    def forward(self, img, canvas):
        """
        prediction
        """
        b, _, H, W = img.shape
        img_feat = self.enc_img(img)
        canvas_feat = self.enc_canvas(canvas)
        h, w = img_feat.shape[-2:]
        feat = paddle.concat([img_feat, canvas_feat], axis=1)
        feat_conv = self.conv(feat)

        pos_embed = paddle.concat([
            self.col_embed[:w].unsqueeze(0).tile([h, 1, 1]),
            self.row_embed[:h].unsqueeze(1).tile([1, w, 1]),
        ], axis=-1).flatten(0, 1).unsqueeze(1)

        hidden_state = self.transformer((pos_embed + feat_conv.flatten(2).transpose([2, 0, 1])).transpose([1, 0, 2]),
                                        self.query_pos.unsqueeze(1).tile([1, b, 1]).transpose([1, 0, 2]))

        param = self.linear_param(hidden_state)
        decision = self.linear_decider(hidden_state)
        return param, decision