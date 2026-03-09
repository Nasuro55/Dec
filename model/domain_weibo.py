import logging
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import BertModel, CLIPModel
from timm.models.vision_transformer import Block

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Loss Functions and Mathematical Utilities
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
    """Hilbert-Schmidt Independence Criterion (HSIC) for feature decoupling"""
    N = X.shape[0]
    if N < 2: return torch.tensor(0.0, device=X.device)
    Kx = rbf_kernel(X, X)
    Ky = rbf_kernel(Y, Y)
    H = torch.eye(N, device=X.device) - torch.ones((N, N), device=X.device) / N
    hsic = torch.trace(torch.mm(Kx, torch.mm(H, torch.mm(Ky, H)))) / ((N - 1) ** 2)
    return hsic

def mmd_loss(X, Y):
    """Maximum Mean Discrepancy (MMD) for distribution consensus alignment"""
    xx, yy, zz = torch.mm(X, X.t()), torch.mm(Y, Y.t()), torch.mm(X, Y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (YY.diag().unsqueeze(0).expand_as(yy))
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
    """Sinkhorn algorithm approximation for Optimal Transport (OT)"""
    B, K, _ = C.shape
    mu = torch.empty(B, K, dtype=torch.float, requires_grad=False, device=C.device).fill_(1.0 / K)
    nu = torch.empty(B, K, dtype=torch.float, requires_grad=False, device=C.device).fill_(1.0 / K)
    
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    
    # Prevent NaN by limiting C range
    C = C / (C.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
    
    for i in range(n_iters):
        u = epsilon * (torch.log(mu) - torch.logsumexp(-C/epsilon + v.unsqueeze(1), dim=-1)) + v
        v = epsilon * (torch.log(nu) - torch.logsumexp(-C.transpose(-2,-1)/epsilon + u.unsqueeze(1), dim=-1)) + u
        
    T = torch.exp((-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon)
    return T

# ==========================================
# Model Dependencies and Layers
# ==========================================

try:
    import models_mae
except ImportError:
    logger.error("Could not import models_mae. Ensure models_mae.py is in the root directory.")
    raise

try:
    from utils.utils_gossipcop import clipdata2gpu, Averager, calculate_metrics, Recorder
except ImportError as e:
    logger.error(f"Failed to import from utils: {e}")
    raise

try:
    from .layers import *
    from .pivot import *
except ImportError:
    logger.warning("Custom .layers or .pivot not found. Placeholders will be used.")

class SimpleGate(nn.Module):
    def __init__(self, dim=1): super(SimpleGate, self).__init__(); self.dim = dim
    def forward(self, x): x1, x2 = x.chunk(2, dim=self.dim); return x1 * x2

class AdaIN(nn.Module):
    def __init__(self): super().__init__()
    def mu(self, x):
        if x is None: return None
        if x.dim() == 3: return torch.mean(x, dim=1)
        elif x.dim() == 2: return torch.mean(x, dim=0, keepdim=True)
        else: return torch.mean(x)

    def sigma(self, x):
        if x is None: return None
        if x.dim() == 3:
            mu_val = self.mu(x).unsqueeze(1)
            return torch.sqrt(torch.mean((x - mu_val)**2, dim=1) + 1e-8)
        elif x.dim() == 2:
            return torch.sqrt(torch.mean((x - self.mu(x))**2, dim=0, keepdim=True) + 1e-8)
        else: return torch.std(x) + 1e-8

    def forward(self, x, mu, sigma):
        if x is None or mu is None or sigma is None: return x
        x_dim = x.dim()
        x_mean = self.mu(x)
        x_std = self.sigma(x)

        if x_dim == 3:
            if x_mean.dim() == 2: x_mean = x_mean.unsqueeze(1)
            if x_std.dim() == 2: x_std = x_std.unsqueeze(1)
        x_norm = (x - x_mean) / (x_std + 1e-8)
        if mu.dim() == 2 and x_norm.dim() == 3: mu = mu.unsqueeze(1)
        if sigma.dim() == 2 and x_norm.dim() == 3: sigma = sigma.unsqueeze(1)
        sigma = torch.relu(sigma) + 1e-8
        return sigma * x_norm + mu

class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims,
                 bert_path_or_name,
                 clip_path_or_name,
                 out_channels, dropout, use_cuda=True,
                 text_token_len=197, image_token_len=197):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.use_cuda = use_cuda; self.num_expert = 6; self.domain_num = 2; self.num_share = 1
        self.unified_dim = 768; self.text_dim = 768; self.image_dim = 768
        self.text_token_len_expected = text_token_len; self.image_token_len_expected = image_token_len + 1
        self.bert_path = bert_path_or_name; self.clip_path = clip_path_or_name

        # Initialize BERT
        try:
            logger.info(f"Loading BERT: {self.bert_path}")
            self.bert = BertModel.from_pretrained(self.bert_path, local_files_only=True)
            for p in self.bert.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.bert = self.bert.cuda()
        except Exception as e:
            logger.error(f"Failed BERT load: {e}")
            self.bert = None

        # Initialize MAE
        self.model_size = "base"; mae_cp = f'./mae_pretrain_vit_{self.model_size}.pth'
        try:
            self.image_model = models_mae.__dict__[f"mae_vit_{self.model_size}_patch16"](norm_pix_loss=False)
            if os.path.exists(mae_cp):
                sd = torch.load(mae_cp, map_location='cpu')
                self.image_model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
            for p in self.image_model.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.image_model = self.image_model.cuda()
        except Exception as e:
            logger.exception(f"Failed MAE load: {e}")
            self.image_model = None

        # Initialize CLIP
        try:
            self.clip_model = CLIPModel.from_pretrained(self.clip_path, local_files_only=True)
            for p in self.clip_model.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.clip_model = self.clip_model.cuda()
        except Exception as e:
            logger.error(f"Failed CLIP load: {e}")
            self.clip_model = None

        fk = {1: 320}
        expert_count = self.num_expert
        shared_count = expert_count * 2
        reasoning_emb_dim = 320

        # Experts Architecture
        self.text_experts = nn.ModuleList([nn.ModuleList([cnn_extractor(self.text_dim, fk) for _ in range(expert_count)]) for _ in range(self.domain_num)])
        self.image_experts = nn.ModuleList([nn.ModuleList([cnn_extractor(self.image_dim, fk) for _ in range(expert_count)]) for _ in range(self.domain_num)])
        self.fusion_experts = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(expert_count)]) for _ in range(self.domain_num)])
        self.text_share_expert = nn.ModuleList([nn.ModuleList([cnn_extractor(self.text_dim, fk) for _ in range(shared_count)]) for _ in range(self.num_share)])
        self.image_share_expert = nn.ModuleList([nn.ModuleList([cnn_extractor(self.image_dim, fk) for _ in range(shared_count)]) for _ in range(self.num_share)])
        self.fusion_share_expert = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(shared_count)]) for _ in range(self.num_share)])

        # Gates and Attention
        gate_out_dim = expert_count + shared_count
        self.image_gate_list = nn.ModuleList([nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(), nn.Linear(self.unified_dim, gate_out_dim), nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in range(self.domain_num)])
        self.text_gate_list = nn.ModuleList([nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(), nn.Linear(self.unified_dim, gate_out_dim), nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in range(self.domain_num)])
        self.fusion_gate_list0 = nn.ModuleList([nn.Sequential(nn.Linear(320, 160), nn.SiLU(), nn.Linear(160, expert_count * 3), nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in range(self.domain_num)])

        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)

        # Classifiers and Fusion Modules
        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.image_classifier = MLP(320, mlp_dims, dropout)
        self.fusion_classifier = MLP(320, mlp_dims, dropout)
        self.max_classifier = MLP(320, mlp_dims, dropout)

        h_dims = mlp_dims if mlp_dims else [348]
        self.MLP_fusion = MLP_fusion(960, 320, h_dims, dropout)
        self.domain_fusion = MLP_fusion(320, 320, h_dims, dropout)
        self.MLP_fusion0 = MLP_fusion(768*2, 768, h_dims, dropout)
        self.clip_fusion = MLP_fusion(1024, 320, h_dims, dropout)

        self.att_mlp_text = MLP_fusion(320, 2, [174], dropout)
        self.att_mlp_img = MLP_fusion(320, 2, [174], dropout)
        self.att_mlp_mm = MLP_fusion(320, 2, [174], dropout)

        # Refinement Modules (output dim 320)
        refinement_input_dim = 320 * 3
        self.refinement_module_text = MLP_fusion(refinement_input_dim, reasoning_emb_dim, [refinement_input_dim // 2], dropout)
        self.refinement_module_image = MLP_fusion(refinement_input_dim, reasoning_emb_dim, [refinement_input_dim // 2], dropout)
        self.refinement_module_cross = MLP_fusion(refinement_input_dim, reasoning_emb_dim, [refinement_input_dim // 2], dropout)

        # Conflict Prototype Space Configuration
        self.num_prototypes = 12 
        self.dim_proto = 320
        self.text_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.dim_proto))
        self.image_prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.dim_proto))
        
        # Cross-Modal Dissonance Modeler
        self.dissonance_mlp = nn.Sequential(
            nn.Linear(self.num_prototypes * self.num_prototypes, 128),
            nn.GELU(),
            nn.Linear(128, self.dim_proto)
        )
        self.final_dissonance_fusion = MLP_fusion(self.dim_proto * 2, self.dim_proto, [self.dim_proto], dropout)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        image_for_mae = kwargs['image']
        clip_pixel_values = kwargs['clip_image']
        clip_input_ids = kwargs['clip_text']
        clip_attention_mask = kwargs.get('clip_attention_mask', None)
        batch_size = inputs.shape[0]
        device = inputs.device

        # --- Modality Encoding ---
        if self.bert:
            bert_outputs = self.bert(input_ids=inputs, attention_mask=masks)
            text_feature_seq = bert_outputs.last_hidden_state
        else:
            text_feature_seq = torch.zeros(batch_size, self.text_token_len_expected, self.unified_dim, device=device)

        if self.image_model:
            image_feature_seq = self.image_model.forward_ying(image_for_mae)
        else:
            image_feature_seq = torch.zeros(batch_size, self.image_token_len_expected, self.unified_dim, device=device)

        if self.clip_model:
            with torch.no_grad():
                clip_img_out = self.clip_model.get_image_features(pixel_values=clip_pixel_values)
                clip_image_embed = clip_img_out / (clip_img_out.norm(dim=-1, keepdim=True) + 1e-8)
                clip_txt_out = self.clip_model.get_text_features(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
                clip_text_embed = clip_txt_out / (clip_txt_out.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            clip_image_embed = torch.zeros(batch_size, 512, device=device)
            clip_text_embed = torch.zeros(batch_size, 512, device=device)

        # Attention and Gating
        text_atn_feature = self.text_attention(text_feature_seq, masks)
        image_atn_feature, _ = self.image_attention(image_feature_seq)
        clip_fusion_feature = self.clip_fusion(torch.cat((clip_image_embed, clip_text_embed), dim=-1).float())
        
        domain_idx = 0
        text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)
        image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)

        # --- Text Experts Calculation ---
        text_experts_feature_sum = torch.zeros((batch_size, 320), device=device)
        text_gate_share_expert_value_sum = torch.zeros((batch_size, 320), device=device)
        for j in range(self.num_expert):
            text_experts_feature_sum += (self.text_experts[domain_idx][j](text_feature_seq) * text_gate_out[:, j].unsqueeze(1))
        for j in range(self.num_expert * 2):
            tmp = self.text_share_expert[0][j](text_feature_seq) * text_gate_out[:, (self.num_expert + j)].unsqueeze(1)
            text_experts_feature_sum += tmp
            text_gate_share_expert_value_sum += tmp
        att_text = F.softmax(self.att_mlp_text(text_experts_feature_sum), dim=-1)
        text_final_features = att_text[:, 0].unsqueeze(1) * text_experts_feature_sum

        # --- Image Experts Calculation ---
        image_experts_feature_sum = torch.zeros((batch_size, 320), device=device)
        image_gate_share_expert_value_sum = torch.zeros((batch_size, 320), device=device)
        for j in range(self.num_expert):
            image_experts_feature_sum += (self.image_experts[domain_idx][j](image_feature_seq) * image_gate_out[:, j].unsqueeze(1))
        for j in range(self.num_expert * 2):
            tmp = self.image_share_expert[0][j](image_feature_seq) * image_gate_out[:, (self.num_expert + j)].unsqueeze(1)
            image_experts_feature_sum += tmp
            image_gate_share_expert_value_sum += tmp
        att_img = F.softmax(self.att_mlp_img(image_experts_feature_sum), dim=-1)
        image_final_features = att_img[:, 0].unsqueeze(1) * image_experts_feature_sum

        # --- Fusion Processing ---
        fusion_share_feature = self.MLP_fusion(torch.cat((clip_fusion_feature, text_gate_share_expert_value_sum, image_gate_share_expert_value_sum), dim=-1))
        fusion_gate_out0 = self.fusion_gate_list0[domain_idx](self.domain_fusion(fusion_share_feature))
        fusion_experts_feature_sum = torch.zeros((batch_size, 320), device=device)
        for n in range(self.num_expert):
            fusion_experts_feature_sum += (self.fusion_experts[domain_idx][n](fusion_share_feature) * fusion_gate_out0[:, n].unsqueeze(1))
        for n in range(self.num_expert * 2):
            fusion_experts_feature_sum += (self.fusion_share_expert[0][n](fusion_share_feature) * fusion_gate_out0[:, self.num_expert + n].unsqueeze(1))
        att_mm = F.softmax(self.att_mlp_mm(fusion_experts_feature_sum), dim=-1)
        fusion_final_features = att_mm[:, 0].unsqueeze(1) * fusion_experts_feature_sum

        # --- Conflict and Dissonance Modeling (OT) ---
        dist_t = torch.cdist(text_final_features, self.text_prototypes) 
        dist_i = torch.cdist(image_final_features, self.image_prototypes)
        C_matrix = torch.cdist(self.text_prototypes, self.image_prototypes).unsqueeze(0).expand(batch_size, -1, -1)
        T_plan = sinkhorn(C_matrix, epsilon=0.05, n_iters=20)
        
        L_conf_model = torch.sum(T_plan * C_matrix, dim=(1,2)).mean() + 0.1 * (dist_t.mean() + dist_i.mean())
        H_diss = self.dissonance_mlp((T_plan * C_matrix).view(batch_size, -1))

        # Final Classification
        text_logits = self.text_classifier(text_final_features).squeeze(-1)
        image_logits = self.image_classifier(image_final_features).squeeze(-1)
        fusion_logits = self.fusion_classifier(fusion_final_features).squeeze(-1)
        
        combined_feat = self.final_dissonance_fusion(torch.cat([text_final_features + image_final_features + fusion_final_features, H_diss], dim=-1))
        final_logits = self.max_classifier(combined_feat).squeeze(-1)

        # Prediction Refinement
        student_concat = torch.cat((text_final_features, image_final_features, fusion_final_features), dim=1)
        pred_refine_text = self.refinement_module_text(student_concat)
        pred_refine_image = self.refinement_module_image(student_concat)
        pred_refine_cross = self.refinement_module_cross(student_concat)

        return (final_logits, text_logits, image_logits, fusion_logits, 
                text_final_features, image_final_features, fusion_final_features,
                pred_refine_text, pred_refine_image, pred_refine_cross,
                text_gate_share_expert_value_sum, image_gate_share_expert_value_sum, L_conf_model)

class Trainer():
    def __init__(self, emb_dim, mlp_dims, bert_path_or_name, clip_path_or_name,
                 use_cuda, lr, dropout, train_loader, val_loader, test_loader, 
                 category_dict, weight_decay, save_param_dir, distillation_weight=0.5, 
                 early_stop=10, epoches=100, metric_key_for_early_stop='acc'):
        self.lr = lr; self.weight_decay = weight_decay; self.train_loader = train_loader
        self.test_loader = test_loader; self.val_loader = val_loader; self.early_stop = early_stop
        self.epoches = epoches; self.category_dict = category_dict; self.use_cuda = use_cuda
        self.save_param_dir = save_param_dir; self.distillation_weight = distillation_weight
        self.metric_key_for_early_stop = metric_key_for_early_stop
        os.makedirs(self.save_param_dir, exist_ok=True)

        # Loss weights
        self.gamma_dec = 0.1  # Decoupling penalty (HSIC)
        self.alpha_conf = 0.1 # Conflict mining penalty (OT)
        self.beta_cec = 0.1   # Consensus calibration penalty (MMD + Moment)

        self.model = MultiDomainPLEFENDModel(emb_dim, mlp_dims, bert_path_or_name, clip_path_or_name, 320, dropout, use_cuda)
        if self.use_cuda: self.model = self.model.cuda()

    def train(self):
        distil_loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop, metric_key=self.metric_key_for_early_stop)

        for epoch in range(self.epoches):
            self.model.train(); avg_loss = Averager()
            train_data_iter = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            for step_n, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu(batch)
                if batch_data is None: continue
                label = batch_data.get('label')
                
                # Forward Pass
                (final_logits, text_logits, image_logits, fusion_logits, 
                 s_text_f, s_image_f, s_fusion_f, p_ref_t, p_ref_i, p_ref_c,
                 t_com, i_com, L_conf) = self.model(**batch_data)

                # 1. Classification Loss
                l_cls = loss_fn(final_logits, label.float()) + (loss_fn(text_logits, label.float()) + loss_fn(image_logits, label.float()) + loss_fn(fusion_logits, label.float())) / 3.0

                # 2. Iterative Calibration Distillation Loss
                l_distil = torch.tensor(0.0, device=final_logits.device)
                t_reas = batch_data.get('text_reasoning_embedding')
                i_reas = batch_data.get('image_reasoning_embedding')
                c_reas = batch_data.get('cross_modal_reasoning_embedding')
                if t_reas is not None:
                    l_distil = distil_loss_fn(p_ref_t, (t_reas - s_text_f).float()) + \
                               distil_loss_fn(p_ref_i, (i_reas - s_image_f).float()) + \
                               distil_loss_fn(p_ref_c, (c_reas - s_fusion_f).float())

                # 3. Decoupling and Alignment (HSIC + MMD + Moment)
                L_dec = hsic_loss(s_text_f, t_com) + hsic_loss(s_image_f, i_com)
                L_mmd = mmd_loss(t_com, i_com)
                m_diff = torch.mean((t_com.mean(0) - i_com.mean(0)) ** 2)
                v_diff = torch.mean((t_com.var(0) - i_com.var(0)) ** 2)
                L_sem = m_diff + v_diff
                
                deco_loss = (self.gamma_dec * L_dec) + (self.alpha_conf * L_conf) + (self.beta_cec * (L_mmd + L_sem))
                
                # Total Loss
                loss = l_cls + (self.distillation_weight * l_distil) + deco_loss
                
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                avg_loss.add(loss.item())
                train_data_iter.set_postfix(loss=avg_loss.item())

            if scheduler: scheduler.step()
            
            # Validation
            val_results = self.test(self.val_loader)
            if val_results:
                mark = recorder.add(val_results)
                if mark == 'save':
                    torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, 'best_model.pth'))
                elif mark == 'esc': break

        return self.test(self.test_loader), os.path.join(self.save_param_dir, 'best_model.pth')

    def test(self, dataloader):
        if dataloader is None: return {}
        self.model.eval(); pred_probs, labels, categories = [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Testing"):
                batch_data = clipdata2gpu(batch)
                if batch_data is None: continue
                logits = self.model(**batch_data)[0]
                pred_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels.extend(batch_data['label'].cpu().numpy().tolist())
                categories.extend(batch_data.get('category', torch.zeros_like(batch_data['label'])).cpu().numpy().tolist())
        return calculate_metrics(labels, pred_probs, categories, self.category_dict) if self.category_dict else calculate_metrics(labels, pred_probs)