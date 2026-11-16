import torch
from torch import nn


class TemporalResidualModule(nn.Module):

    def __init__(self, channels, kernel_size=15, dropout=0.25):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False), nn.Conv1d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False), nn.Conv1d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm1d(channels), nn.Dropout(dropout))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class AttentionPoolingModule(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)

class EEGFeatureModule(nn.Module):

    def __init__(self, n_channels=32, d_model=96, dropout=0.35):
        super().__init__()
        self.temporal_stem = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12), bias=False), nn.BatchNorm2d(32), nn.ELU(), nn.Conv2d(32, 32, kernel_size=(1, 51), padding=(0, 25), groups=32, bias=False), nn.Conv2d(32, 48, kernel_size=(1, 1), bias=False), nn.BatchNorm2d(48), nn.ELU())
        self.spatial_filter = nn.Sequential(nn.Conv2d(48, 96, kernel_size=(n_channels, 1), groups=48, bias=False), nn.BatchNorm2d(96), nn.ELU(), nn.AvgPool2d(kernel_size=(1, 4)), nn.Dropout(dropout))
        self.temporal_blocks = nn.Sequential(TemporalResidualModule(96, kernel_size=15, dropout=dropout * 0.7), TemporalResidualModule(96, kernel_size=31, dropout=dropout * 0.7), nn.AvgPool1d(kernel_size=2))
        self.proj = nn.Sequential(nn.Conv1d(96, d_model, kernel_size=1, bias=False), nn.BatchNorm1d(d_model), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.feature_dim = d_model

    def encode_sequence(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_stem(x)
        x = self.spatial_filter(x).squeeze(2)
        x = self.temporal_blocks(x)
        x = self.proj(x).transpose(1, 2)
        return self.norm(self.transformer(x))

    def forward(self, x):
        x = self.encode_sequence(x)
        return x.mean(dim=1)

class EEGModule(nn.Module):

    def __init__(self, n_channels=32, d_model=96, dropout=0.4):
        super().__init__()
        self.base = EEGFeatureModule(n_channels=n_channels, d_model=d_model, dropout=dropout)
        self.feedback_pool = AttentionPoolingModule(d_model)
        self.fusion = nn.Sequential(nn.Linear(d_model * 5, d_model * 2), nn.LayerNorm(d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))
        self.feature_dim = d_model

    def forward(self, x):
        seq = self.base.encode_sequence(x)
        n_steps = seq.shape[1]
        early_end = max(1, n_steps // 3)
        mid_start = max(0, n_steps // 3)
        late_start = min(n_steps - 1, n_steps // 2)
        global_feat = seq.mean(dim=1)
        early_feat = seq[:, :early_end].mean(dim=1)
        mid_feat = seq[:, mid_start:late_start].mean(dim=1) if late_start > mid_start else global_feat
        late_seq = seq[:, late_start:]
        late_feat = self.feedback_pool(late_seq)
        change_feat = late_feat - early_feat
        return self.fusion(torch.cat([global_feat, early_feat, mid_feat, late_feat, change_feat], dim=1))

class Ours(nn.Module):

    def __init__(self, audio_encoder, eeg_encoder, freeze_audio_encoder=True, fusion_dim=128, n_target=4):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.eeg_encoder = eeg_encoder
        self.freeze_audio_encoder = freeze_audio_encoder
        if freeze_audio_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        audio_dim = getattr(audio_encoder.config, 'hidden_size', None)
        if audio_dim is None:
            audio_dim = getattr(audio_encoder.config, 'd_model')
        eeg_dim = eeg_encoder.feature_dim
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.eeg_proj = nn.Sequential(nn.Linear(eeg_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.audio_head = nn.Sequential(nn.Linear(fusion_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, n_target))
        self.eeg_head = nn.Sequential(nn.Linear(fusion_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, n_target))
        reliability_feature_dim = 7
        self.gate_mlp = nn.Sequential(nn.Linear(reliability_feature_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1))

    def encode_audio(self, encoder_inputs):
        if self.freeze_audio_encoder:
            self.audio_encoder.eval()
            with torch.no_grad():
                outputs = self.audio_encoder(**encoder_inputs)
        else:
            outputs = self.audio_encoder(**encoder_inputs)
        return outputs.last_hidden_state.mean(dim=1)

    @staticmethod
    def entropy(prob):
        return -(prob * prob.clamp_min(1e-08).log()).sum(dim=1, keepdim=True)

    @staticmethod
    def top_margin(prob):
        top2 = torch.topk(prob, k=2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).unsqueeze(1)

    def forward(self, eeg, encoder_inputs):
        audio_feat = self.audio_proj(self.encode_audio(encoder_inputs))
        eeg_feat = self.eeg_proj(self.eeg_encoder(eeg))
        audio_logits = self.audio_head(audio_feat)
        eeg_logits = self.eeg_head(eeg_feat)
        p_audio = torch.softmax(audio_logits, dim=1)
        p_eeg = torch.softmax(eeg_logits, dim=1)
        disagreement = (p_audio.argmax(dim=1) != p_eeg.argmax(dim=1)).float().unsqueeze(1)
        reliability_features = torch.cat([p_audio.max(dim=1, keepdim=True).values, p_eeg.max(dim=1, keepdim=True).values, self.entropy(p_audio), self.entropy(p_eeg), self.top_margin(p_audio), self.top_margin(p_eeg), disagreement], dim=1)
        gate_logit = self.gate_mlp(reliability_features).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logit).unsqueeze(1)
        p_final = (1.0 - gate_prob) * p_audio + gate_prob * p_eeg
        final_logits = torch.log(p_final.clamp_min(1e-08))
        return {'final_logits': final_logits, 'audio_logits': audio_logits, 'eeg_logits': eeg_logits, 'gate_logit': gate_logit, 'gate_prob': gate_prob.squeeze(1)}
