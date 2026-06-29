import torch
import torch.nn as nn
import math

class GMMPrior(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        """
        Module tính toán Loss dựa trên GMM Prior thay thế cho standard Gaussian.
        
        Args:
            latent_dim (int): Số chiều của không gian tiềm ẩn z.
            n_clusters (int): Số lượng cụm tế bào dự kiến (K).
        """
        super(GMMPrior, self).__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        # Khởi tạo các tham số học được của GMM (Bước M sẽ được tối ưu qua Gradient Descent)
        # pi: Xác suất tiên nghiệm của mỗi cụm (phân phối Categorical)
        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters)
        # mu_c: Kỳ vọng của mỗi cụm
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))
        # var_c: Phương sai của mỗi cụm (lưu ý: để tối ưu tốt hơn, thường giữ log_var thay vì var)
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def forward(self, z, mu, logvar):
        """
        Tính toán KL Divergence dựa trên GMM.
        
        Args:
            z (Tensor): Latent vector sau khi reparameterization [Batch_size, latent_dim]
            mu (Tensor): Mean từ Encoder [Batch_size, latent_dim]
            logvar (Tensor): Log variance từ Encoder [Batch_size, latent_dim]
            
        Returns:
            loss_kl_gmm (Tensor): Giá trị Loss KL để cộng vào hàm mục tiêu
            gamma (Tensor): Xác suất điểm dữ liệu thuộc về từng cụm [Batch_size, n_clusters]
        """
        batch_size = z.size(0)
        
        # Đảm bảo pi là một phân phối xác suất hợp lệ (tổng = 1)
        log_pi = torch.log_softmax(self.pi, dim=0)
        var_c = torch.exp(self.logvar_c)

        # ---------------------------------------------------------
        # BƯỚC E (Expectation): Tính xác suất hậu nghiệm gamma_c = p(c|z)
        # ---------------------------------------------------------
        # Mở rộng chiều để broadcasting: [Batch_size, K, latent_dim]
        z_expand = z.unsqueeze(1)
        mu_c_expand = self.mu_c.unsqueeze(0)
        var_c_expand = var_c.unsqueeze(0)

        # Tính log p(z|c)
        log_p_z_c = -0.5 * torch.sum(
            torch.log(2 * math.pi * var_c_expand) + 
            ((z_expand - mu_c_expand) ** 2) / var_c_expand, 
            dim=2
        )

        # Tính log( p(c) * p(z|c) )
        log_p_c_z = log_pi + log_p_z_c
        
        # Chuẩn hóa để lấy xác suất gamma (tránh underflow bằng logsumexp)
        gamma = torch.softmax(log_p_c_z, dim=1) 

        # ---------------------------------------------------------
        # Tính KL Divergence (D_KL) cho VaDE
        # ---------------------------------------------------------
        # 1. Thành phần rời rạc: KL(gamma || pi)
        kl_discrete = torch.sum(gamma * (torch.log(gamma + 1e-10) - log_pi), dim=1)

        # 2. Thành phần liên tục: gamma * KL( N(mu, var) || N(mu_c, var_c) )
        mu_expand = mu.unsqueeze(1)
        logvar_expand = logvar.unsqueeze(1)
        var_expand = torch.exp(logvar_expand)

        kl_continuous = 0.5 * torch.sum(
            gamma * torch.sum(
                self.logvar_c.unsqueeze(0) - logvar_expand + 
                (var_expand + (mu_expand - mu_c_expand) ** 2) / var_c_expand - 1, 
                dim=2
            ), 
            dim=1
        )

        # Tổng hợp Loss (trung bình trên batch)
        loss_kl_gmm = torch.mean(kl_discrete + kl_continuous)

        return loss_kl_gmm, gamma