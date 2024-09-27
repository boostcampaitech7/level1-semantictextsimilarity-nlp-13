import torch.nn as nn
import torch.nn.functional as F
import transformers


class STSModel(nn.Module):
    def __init__(self, plm_name):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True
        )

    def forward(self, x):
        x = self.plm(x)["logits"]
        return x


class NaverConnectModelWithDropout(nn.Module):
    def __init__(self, plm_name):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True
        )
        self.dropout = nn.Dropout(0.3)  # 드롭아웃 레이어 추가

    def forward(self, x):
        x = self.plm(x)["logits"]
        x = self.dropout(x)
        return x


class SevenElevenWithBiLSTM(nn.Module):
    def __init__(self, plm_name, hidden_size=64):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name,
            num_labels=hidden_size,
            use_auth_token=True,
        )
        self.bilstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(
            hidden_size * 2, 1
        )  # 양방향 LSTM이기 때문에 hidden_size * 2

    def forward(self, x):
        x = self.plm(x)["logits"]
        x, _ = self.bilstm(x.unsqueeze(1))
        x = self.fc(x.squeeze(1))
        return x


class STSModelWithAttention(nn.Module):
    def __init__(self, plm_name, hidden_size=128):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name,
            num_labels=hidden_size,
            use_auth_token=True,
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.plm(x)["logits"]
        x, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = self.fc(x.squeeze(0))
        return x


class STSModelWithResidualConnection(nn.Module):
    def __init__(self, plm_name):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name,
            num_labels=1,
            use_auth_token=True,
        )
        self.residual_fc = nn.Linear(1, 1)

    def forward(self, x):
        plm_output = self.plm(x)["logits"]
        residual = self.residual_fc(plm_output)  # Residual Connection 적용
        x = plm_output + residual  # Residual 연결
        return x
