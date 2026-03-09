import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer, AutoModel
import models_mae 
from utils.utils_weibo import clipdata2gpu, Averager, metricsTrueFalse, Recorder
from .layers import MLP, MaskAttention, TokenAttention, cnn_extractor, LayerNorm, MLP_Mu, MLP_fusion, clip_fuion
from .pivot import TransformerLayer, MLP_trans
from timm.models.vision_transformer import Block

try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name
except ImportError:
    print("Warning: cn_clip library not found. CLIP functionalities will not work.")
    clip = None
    load_from_name = None

# ==========================================
# Core Mathematical Modules for DeCo
# ==========================================

def rbf_kernel(X, Y, sigma=1.0):
    """Radial Basis Function (RBF) Kernel"""
    XX = X.matmul(X.t())
    YY = Y.matmul(Y.t())
    XY = X.matmul(Y.t())
    rx = (XX.diag().unsqueeze(0).expand_as(XX))
    ry = (YY.diag().unsqueeze(0).expand_as(YY))
    K = torch.exp(- (rx.t() + ry - 2*XY) / (2 * sigma**2))
    return K

def hsic_loss(X, Y):
    """Hilbert-Schmidt Independence Criterion for feature decoupling (Eq.3)"""
    N = X.shape[0]
    if N < 2: return torch.tensor(0.0, device=X.device)
    Kx = rbf_kernel(X, X)
    Ky = rbf_kernel(Y, Y)
    H = torch.eye(N, device=X.device) - torch.ones((N, N), device=X.device) / N
    hsic = torch.trace(torch.mm(Kx, torch.mm(H, torch.mm(Ky, H)))) / ((N - 1) ** 2)
    return hsic

def mmd_loss(X, Y):
    """Maximum Mean Discrepancy for distribution consensus alignment (Eq.11)"""
    xx, yy, zz = torch.mm(X, X.t()), torch.mm(Y, Y.t()), torch.mm(X, Y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz
    XX, YY, XY = (torch.zeros_like(xx), torch.zeros_like(xx), torch.zeros_like(xx))
    for a in [0.5, 1, 2]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1
    return torch.mean(XX + YY - 2.*XY)

def sinkhorn(C, epsilon=0.05, n_iters=20):
    """Sinkhorn algorithm for Optimal Transport approximation (Eq.7)"""
    B, K, _ = C.shape
    mu = torch.empty(B, K, dtype=torch.float, requires_grad=False, device=C.device).fill_(1.0 / K)
    nu = torch.empty(B, K, dtype=torch.float, requires_grad=False, device=C.device).fill_(1.0 / K)
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    
    # Normalize Cost matrix to prevent NaN
    C = C / (C.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
    
    for i in range(n_iters):
        u = epsilon * (torch.log(mu) - torch.logsumexp(-C/epsilon + v.unsqueeze(1), dim=-1)) + v
        v = epsilon * (torch.log(nu) - torch.logsumexp(-C.transpose(-2,-1)/epsilon + u.unsqueeze(1), dim=-1)) + u
        
    T = torch.exp((-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon)
    return T

class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x,(1))/(x.shape[1] + 1e-6)

    def sigma(self, x):
        return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1] + 1e-6))

    def forward(self, x, mu, sigma):
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean/(x_std + 1e-6)
        return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])

class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout,
                 reasoning_emb_dim=768, num_manipulation_classes=0):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.num_expert = 6
        self.task_num = 2
        self.domain_num = self.task_num
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768  # MAE ViT-Base output

        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        # Text Experts
        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert)]
            text_expert_list.append(nn.ModuleList(text_expert))
        self.text_experts = nn.ModuleList(text_expert_list)

        # Image Experts
        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert)]
            image_expert_list.append(nn.ModuleList(image_expert))
        self.image_experts = nn.ModuleList(image_expert_list)

        # Fusion Experts
        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert)]
            fusion_expert_list.append(nn.ModuleList(fusion_expert))
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        # Final Refinement Experts
        final_expert_list = []
        for i in range(self.domain_num):
            final_expert = [Block(dim=320, num_heads=8) for _ in range(self.num_expert)]
            final_expert_list.append(nn.ModuleList(final_expert))
        self.final_experts = nn.ModuleList(final_expert_list)

        # Shared Expert Components
        text_share_expert_outer, image_share_expert_outer, fusion_share_expert_outer, final_share_expert_outer = [], [], [], []
        for _ in range(self.num_share):
            text_share = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert * 2)]
            image_share = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert * 2)]
            fusion_share = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert*2)]
            final_share = [Block(dim=320, num_heads=8) for _ in range(self.num_expert*2)]

            text_share_expert_outer.append(nn.ModuleList(text_share))
            image_share_expert_outer.append(nn.ModuleList(image_share))
            fusion_share_expert_outer.append(nn.ModuleList(fusion_share))
            final_share_expert_outer.append(nn.ModuleList(final_share))

        self.text_share_expert = nn.ModuleList(text_share_expert_outer)
        self.image_share_expert = nn.ModuleList(image_share_expert_outer)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert_outer)
        self.final_share_expert = nn.ModuleList(final_share_expert_outer)

        # Expert Gating Networks
        gate_output_dim_specific_plus_shared = self.num_expert * 3
        gate_output_dim_fusion_original = self.num_expert * 4

        image_gate_list, text_gate_list, fusion_gate_list_original, fusion_gate_list0, final_gate_list = [], [], [], [], []
        for _ in range(self.domain_num):
            text_gate_list.append(nn.Sequential(
                nn.Linear(self.text_dim, self.text_dim), nn.SiLU(),
                nn.Linear(self.text_dim, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
            ))
            image_gate_list.append(nn.Sequential(
                nn.Linear(self.image_dim, self.image_dim), nn.SiLU(),
                nn.Linear(self.image_dim, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
            ))
            fusion_gate_list_original.append(nn.Sequential(
                nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(),
                nn.Linear(self.unified_dim, gate_output_dim_fusion_original), nn.Dropout(0.1), nn.Softmax(dim=1)
            ))
            fusion_gate_list0.append(nn.Sequential(
                nn.Linear(320, 160), nn.SiLU(),
                nn.Linear(160, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
            ))
            final_gate_list.append(nn.Sequential(
                nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 160), nn.SiLU(),
                nn.Linear(160, gate_output_dim_specific_plus_shared), nn.Dropout(0.1), nn.Softmax(dim=1)
            ))

        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list_original)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        self.final_gate_list = nn.ModuleList(final_gate_list)

        # Cross-Modal Attention Layers
        self.text_attention = MaskAttention(self.text_dim)
        self.image_attention = TokenAttention(self.image_dim)
        self.fusion_attention = TokenAttention(self.text_dim + self.image_dim)
        self.final_attention = TokenAttention(320)

        # Output Classifiers
        feature_dim_after_experts = 320
        self.text_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.image_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.fusion_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.max_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)

        final_classifier_list = []
        for i in range(self.domain_num):
            final_classifier_list.append(MLP(feature_dim_after_experts, mlp_dims, dropout))
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        # Feature Projection and Fusion MLPs
        self.MLP_fusion = MLP_fusion(320 * 3, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(self.text_dim + self.image_dim, self.text_dim, [348], 0.1)

        if clip is not None:
             self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
        else:
            self.clip_fusion = None

        self.att_mlp_text = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
        self.att_mlp_img = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
        self.att_mlp_mm = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)

        # MAE Backbone Initialization
        self.model_size = "base"
        try:
            self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
            checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
            self.image_model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available():
                self.image_model.cuda()
            for param in self.image_model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load MAE model. Error: {e}")
            self.image_model = None

        # CLIP Backbone Initialization
        if clip is not None:
            try:
                clip_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.ClipModel, _ = load_from_name("ViT-B-16", device=clip_device, download_root='./')
            except Exception as e:
                print(f"Warning: Could not load CLIP model. Error: {e}")
                self.ClipModel = None
        else:
            self.ClipModel = None

        # Enhanced Calibration Distillation Modules
        self.reasoning_emb_dim = reasoning_emb_dim
        self.num_manipulation_classes = num_manipulation_classes
        student_feature_dim = 320
        
        projection_hidden_dim = (student_feature_dim + self.reasoning_emb_dim) // 2
        self.direct_projection_text = MLP_fusion(student_feature_dim, self.reasoning_emb_dim, [projection_hidden_dim], dropout)
        self.direct_projection_image = MLP_fusion(student_feature_dim, self.reasoning_emb_dim, [projection_hidden_dim], dropout)
        self.direct_projection_cross = MLP_fusion(student_feature_dim, self.reasoning_emb_dim, [projection_hidden_dim], dropout)

        refinement_input_dim = self.reasoning_emb_dim * 3
        refinement_hidden_dim = (refinement_input_dim + self.reasoning_emb_dim) // 2
        self.refinement_module_text = MLP_fusion(refinement_input_dim, self.reasoning_emb_dim, [refinement_hidden_dim], dropout)
        self.refinement_module_image = MLP_fusion(refinement_input_dim, self.reasoning_emb_dim, [refinement_hidden_dim], dropout)
        self.refinement_module_cross = MLP_fusion(refinement_input_dim, self.reasoning_emb_dim, [refinement_hidden_dim], dropout)
        
        if self.num_manipulation_classes > 0:
            self.manipulation_classifier = MLP(student_feature_dim, [self.num_manipulation_classes], dropout)
        else:
            self.manipulation_classifier = None

        # DeCo Specific Components (Prototype Space & Dissonance Modeler)
        self.num_prototypes = 12 # K=12 as defined in the paper
        self.dim_proto = 320
        
        # Conflict Prototype Space (Eq.4)
        self.text_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.dim_proto))
        self.image_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.dim_proto))
        
        # Cross-Modal Dissonance Modeler (Eq.12)
        self.dissonance_mlp = nn.Sequential(
            nn.Linear(self.num_prototypes * self.num_prototypes, 128),
            nn.GELU(),
            nn.Linear(128, self.dim_proto)
        )
        self.final_dissonance_fusion = MLP_fusion(self.dim_proto * 2, self.dim_proto, [self.dim_proto], dropout)

    def get_expert_output(self, features, gate_outputs, specific_experts, shared_experts, is_final_expert=False):
        batch_size = features.size(0)
        num_specific_experts = len(specific_experts)
        num_shared_experts_total = len(shared_experts[0])

        expert_output_dim = 320
        expert_outputs_sum = torch.zeros(batch_size, expert_output_dim, device=features.device)
        shared_expert_outputs_sum = torch.zeros(batch_size, expert_output_dim, device=features.device)

        for i in range(num_specific_experts):
            expert_out = specific_experts[i](features)
            gate_val = gate_outputs[:, i].unsqueeze(1)
            expert_outputs_sum += expert_out * gate_val

        for i in range(num_shared_experts_total):
            expert_out = shared_experts[0][i](features)
            gate_val = gate_outputs[:, num_specific_experts + i].unsqueeze(1)
            expert_outputs_sum += expert_out * gate_val
            if not is_final_expert:
                shared_expert_outputs_sum += expert_out * gate_val

        return expert_outputs_sum, shared_expert_outputs_sum

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        image_raw = kwargs['image']

        text_feature_full = self.bert(inputs, attention_mask=masks)[0]

        if self.image_model:
            try:
                image_feature_full = self.image_model.forward_ying(image_raw)
            except AttributeError:
                image_feature_full = self.image_model.forward_features(image_raw)
        else:
            image_feature_full = torch.zeros_like(text_feature_full, device=text_feature_full.device)

        clip_image_input = kwargs.get('clip_image')
        clip_text_input = kwargs.get('clip_text')
        clip_fusion_feature = torch.zeros(text_feature_full.size(0), 320, device=text_feature_full.device)
        
        if self.ClipModel and self.clip_fusion and clip_image_input is not None and clip_text_input is not None:
            with torch.no_grad():
                clip_image_feature = self.ClipModel.encode_image(clip_image_input)
                clip_text_feature = self.ClipModel.encode_text(clip_text_input)
                clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
                clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)
            clip_concat_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1).float()
            clip_fusion_feature = self.clip_fusion(clip_concat_feature)

        text_atn_feature = self.text_attention(text_feature_full, masks)
        image_atn_feature, _ = self.image_attention(image_feature_full)

        domain_idx = kwargs.get('domain_idx', 0) % self.domain_num
        text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)
        image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)

        # Expert processing
        text_experts_output, text_shared_output = self.get_expert_output(
            text_feature_full, text_gate_out, self.text_experts[domain_idx], self.text_share_expert
        )
        text_att_split = F.softmax(self.att_mlp_text(text_experts_output), dim=-1)
        text_final_feature = text_att_split[:, 0].unsqueeze(1) * text_experts_output

        image_experts_output, image_shared_output = self.get_expert_output(
            image_feature_full, image_gate_out, self.image_experts[domain_idx], self.image_share_expert
        )
        image_att_split = F.softmax(self.att_mlp_img(image_experts_output), dim=-1)
        image_final_feature = image_att_split[:, 0].unsqueeze(1) * image_experts_output

        concat_for_MLP_fusion = torch.cat((clip_fusion_feature, text_shared_output, image_shared_output), dim=-1)
        fusion_share_base_feature = self.MLP_fusion(concat_for_MLP_fusion)
        fusion_gate_out0 = self.fusion_gate_list0[domain_idx](self.domain_fusion(fusion_share_base_feature))

        fusion_experts_output, _ = self.get_expert_output(
            fusion_share_base_feature, fusion_gate_out0, self.fusion_experts[domain_idx], self.fusion_share_expert, is_final_expert=True
        )
        fusion_att_split = F.softmax(self.att_mlp_mm(fusion_experts_output), dim=-1)
        fusion_final_feature = fusion_att_split[:, 0].unsqueeze(1) * fusion_experts_output

        # DeCo Logic: Cross-Modal Conflict Mining (OT Matching & Dissonance)
        B = text_final_feature.size(0)
        dist_t = torch.cdist(text_final_feature, self.text_prototypes)
        dist_i = torch.cdist(image_final_feature, self.image_prototypes)
        
        C = torch.cdist(self.text_prototypes, self.image_prototypes).unsqueeze(0).expand(B, -1, -1)
        T_plan = sinkhorn(C, epsilon=0.05, n_iters=20)
        
        conf_cost = torch.sum(T_plan * C, dim=(1,2))
        L_conf_model = conf_cost.mean() + 0.1 * (dist_t.mean() + dist_i.mean())

        # Extracting dissonance representation (H_diss)
        H_diss = self.dissonance_mlp((T_plan * C).view(B, -1))
        all_modality_combined = text_final_feature + image_final_feature + fusion_final_feature
        all_modality_enhanced = self.final_dissonance_fusion(torch.cat([all_modality_combined, H_diss], dim=-1))

        # Predictions
        final_fake_news_prob = torch.sigmoid(self.max_classifier(all_modality_enhanced).squeeze(1))
        text_fake_news_prob = torch.sigmoid(self.text_classifier(text_final_feature).squeeze(1))
        image_fake_news_prob = torch.sigmoid(self.image_classifier(image_final_feature).squeeze(1))
        fusion_fake_news_prob = torch.sigmoid(self.fusion_classifier(fusion_final_feature).squeeze(1))

        # Student-Teacher Distillation features
        S_text = self.direct_projection_text(text_final_feature)
        S_image = self.direct_projection_image(image_final_feature)
        S_cross = self.direct_projection_cross(fusion_final_feature)
        S_concat = torch.cat((S_text, S_image, S_cross), dim=1)
        
        S_final_text = S_text + self.refinement_module_text(S_concat)
        S_final_image = S_image + self.refinement_module_image(S_concat)
        S_final_cross = S_cross + self.refinement_module_cross(S_concat)

        manip_pred_logits = self.manipulation_classifier(all_modality_enhanced) if self.manipulation_classifier else None

        return (final_fake_news_prob, text_fake_news_prob, image_fake_news_prob, fusion_fake_news_prob,
                S_text, S_image, S_cross, S_final_text, S_final_image, S_final_cross,
                manip_pred_logits, text_experts_output, text_shared_output,
                image_experts_output, image_shared_output, L_conf_model)

class DOMAINTrainerWeibo():
    def __init__(self, emb_dim, mlp_dims, bert, use_cuda, lr, dropout,
                 train_loader, val_loader, test_loader, category_dict,
                 weight_decay, save_param_dir, reasoning_emb_dim=768,
                 num_manipulation_classes=0, distillation_weight=0.1,
                 lambda_manipulation_predict=0.1, early_stop=100, epoches=100):
        self.lr, self.weight_decay = lr, weight_decay
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.early_stop, self.epoches = early_stop, epoches
        self.category_dict, self.use_cuda = category_dict, use_cuda
        self.save_param_dir = save_param_dir
        if not os.path.exists(self.save_param_dir): os.makedirs(self.save_param_dir, exist_ok=True)

        self.distillation_weight = distillation_weight
        self.lambda_manipulation_predict = lambda_manipulation_predict
        self.alpha_distill = 0.5 

        # DeCo Joint Optimization Hyperparameters (Eq.13)
        self.gamma_dec = 0.1  # Decoupling penalty (HSIC)
        self.alpha_conf = 0.1 # Conflict mining penalty (OT)
        self.beta_cec = 0.1   # Consensus calibration penalty (MMD + Moment)

        self.model = MultiDomainPLEFENDModel(emb_dim, mlp_dims, bert, 320, dropout, reasoning_emb_dim, num_manipulation_classes)
        if self.use_cuda: self.model = self.model.cuda()

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss() if num_manipulation_classes > 0 else None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.98)
        self.model_save_filename = 'parameter_clip111.pkl'

    def train(self):
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            avg_loss_epoch = Averager()

            for batch in train_data_iter:
                batch_data = clipdata2gpu(batch, self.use_cuda)
                labels = batch_data['label'].float()

                outputs = self.model(**batch_data)
                final_prob, a_t, a_i, a_f = outputs[0:4]
                s_t, s_i, s_c, sf_t, sf_i, sf_c = outputs[4:10]
                manip_logits, text_uni, text_com, img_uni, img_com, L_conf = outputs[10:]

                # Detection Loss
                loss_det = self.bce_loss(final_prob, labels) + (self.bce_loss(a_t, labels) + self.bce_loss(a_i, labels) + self.bce_loss(a_f, labels)) / 3.0

                # Distillation Loss
                loss_distill = torch.tensor(0.0, device=labels.device)
                t_t, t_i, t_c = batch_data.get('teacher_reasoning_text_emb'), batch_data.get('teacher_reasoning_image_emb'), batch_data.get('teacher_reasoning_cross_emb')
                if t_t is not None:
                    l_align = (self.mse_loss(s_t, t_t) + self.mse_loss(s_i, t_i) + self.mse_loss(s_c, t_c)) / 3.0
                    l_refine = (self.mse_loss(sf_t, t_t) + self.mse_loss(sf_i, t_i) + self.mse_loss(sf_c, t_c)) / 3.0
                    loss_distill = 0.5 * l_align + 0.5 * l_refine

                # Manipulation Prediction Loss
                loss_manip = self.ce_loss(manip_logits, batch_data['manipulation_labels'].long()) if self.ce_loss and manip_logits is not None else torch.tensor(0.0)

                # DeCo Specific Losses
                L_dec = hsic_loss(text_uni, text_com) + hsic_loss(img_uni, img_com)
                L_mmd = mmd_loss(text_com, img_com)
                L_sem = torch.mean((text_com.mean(0) - img_com.mean(0))**2) + torch.mean((text_com.var(0) - img_com.var(0))**2)
                L_cec = L_sem + L_mmd

                deco_total_loss = (self.gamma_dec * L_dec) + (self.alpha_conf * L_conf) + (self.beta_cec * L_cec)

                total_loss = loss_det + self.distillation_weight * loss_distill + self.lambda_manipulation_predict * loss_manip + deco_total_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                if self.scheduler: self.scheduler.step()

                avg_loss_epoch.add(total_loss.item())
                train_data_iter.set_postfix(loss=avg_loss_epoch.item())

            val_results = self.test(self.test_loader)
            mark = recorder.add(val_results)
            if mark == 'save': torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, self.model_save_filename))
            elif mark == 'esc': break

        return val_results

    def test(self, dataloader):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                batch_data = clipdata2gpu(batch, self.use_cuda)
                preds.extend(self.model(**batch_data)[0].cpu().numpy().tolist())
                labels.extend(batch_data['label'].cpu().numpy().tolist())
        
        acc = sum(1 for p, l in zip(preds, labels) if (p > 0.5) == l) / len(labels)
        return {"accuracy": acc}