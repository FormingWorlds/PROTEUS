
import torch
from torch import Tensor


from botorch.models.model import Model

from botorch.acquisition import AnalyticAcquisitionFunction

from botorch.optim import optimize_acqf

from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PosteriorSample(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model)

        self.path = draw_matheron_paths(self.model, torch.Size([1]))

    def forward(self, X: Tensor):
        
        # X: N, ..., d

        y = self.path(X)

        return y.squeeze(-1) # N

class PenalizedUpperConfidenceBound(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        bounds: Tensor,
        busy: Tensor | None = None,
        y_max = None,
        local = False
    ) -> None:
        super(AnalyticAcquisitionFunction, self).__init__(model)

        self.register_buffer("beta", torch.as_tensor(beta, dtype=dtype, device = device))
        self.register_buffer("bounds", torch.as_tensor(bounds, dtype=dtype, device = device))
        if y_max is not None:
            self.register_buffer("y_max", torch.as_tensor(y_max, dtype=dtype, device = device))
        
        if busy is not None:

            self.register_buffer("busy", torch.as_tensor(busy, dtype=dtype, device = device))
            grad_norm = AnalyticPostMeanGradientNorm(self.model)

            d = self.bounds.shape[-1]

            if local:
                ls = self.model.covar_module.lengthscale # 1xd

                bounds_l = torch.clamp_min(self.busy-ls, self.bounds[0]) # bxd
                bounds_u = torch.clamp_max(self.busy+ls, self.bounds[1]) # bxd
                bounds_ = torch.stack((bounds_l, bounds_u), dim = 1) # bx2xd
    
                L = []
                norm_maxer = []
                for i in range(len(self.busy)):
                    
                    # this will run on GPU if all inputs are on GPU
                    maxer, l = optimize_acqf(   acq_function=grad_norm,
                                                bounds=bounds_[i],
                                                q=1,
                                                num_restarts=10,
                                                raw_samples=d*1000,
                                                options={"batch_limit": 50, "maxiter": 200},
                                            )

                    L.append(l)
                    norm_maxer.append(maxer)

                L = torch.tensor(L, dtype=dtype, device = device).reshape(1,len(self.busy)) # 1xb
                norm_maxer = torch.cat(norm_maxer).reshape(-1, d) # bxd
            else:

                norm_maxer, L = optimize_acqf(  acq_function=grad_norm,
                                                bounds=self.bounds,
                                                q=1,
                                                num_restarts=10,
                                                raw_samples=d*1000,
                                                options={"batch_limit": 50, "maxiter": 200},
                                            )
                
                L = L.to(dtype=dtype, device = device).reshape(1,1)

            # cheeky fix for flat regions
            # L = torch.where(L < 1e-7, torch.tensor(10, dtype=dtype, device=device), L)

            self.register_buffer("L", torch.as_tensor(L, dtype=dtype, device=device))
            self.register_buffer("norm_maxer", torch.as_tensor(norm_maxer, dtype=dtype))
            
        else:
            self.busy = busy

        
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X (tensor): Nxq=1xd

        Returns:
            acqf value (tensor): N
        """

        p = -5

        self.beta = self.beta
        
        posterior = self.model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        ucb =  (mean + self.beta * std).flatten() # N

        if self.busy is None:
            return ucb 

        post_b = self.model.posterior(self.busy)
        mean_b = post_b.mean # bx1
   
        eps = 1e-8
        std_b = post_b.variance.sqrt() #bx1

        if len(X.shape) == 3: X = X.squeeze(1) # remove q-batch dim
        norm = torch.cdist(X, self.busy) # + 1e-8 # Nxb 

        s = ((torch.abs(mean_b - self.y_max) + 1 * std_b)).reshape(1,-1) / (self.L) # 1xb

        weights = norm / s # Nxb

        diff_weights = (weights**p + 1)**(1/p) # Nxb

        pen = torch.sum(torch.log(diff_weights), dim=1) # N

        pen_ucb = torch.exp(torch.log(ucb.clamp_min(eps)) + pen)

        if torch.any(torch.isnan(pen_ucb)) or torch.any(torch.isinf(pen_ucb)) or torch.all(pen_ucb == 0):
            print(f"Invalid values detected: {pen_ucb}")

        return pen_ucb # N
    
class AnalyticPostMeanGradientNorm(AnalyticAcquisitionFunction):
    def __init__(self, 
                 model: Model,
                 ) -> None:
        super().__init__(model)

        self.k = model.covar_module
        self.Theta_inv = torch.atleast_2d(torch.diag(1/self.k.lengthscale.flatten()**2))
        self.train_X = model.input_transform.untransform(model.train_inputs[0])
        self.train_Y = model.outcome_transform.untransform(model.train_targets)[0].reshape(-1,1)

        K_X_X = self.k(self.train_X).evaluate()
        sig_squ = model.likelihood.noise

        K_noise = K_X_X + (sig_squ + 1e-8) * torch.eye(K_X_X.size(0), dtype=dtype, device=device)

        L = torch.linalg.cholesky(K_noise + 1e-8 * torch.eye(K_X_X.size(0), dtype=dtype, device=device))
        K_noise_inv = torch.cholesky_inverse(L)

        self.K_noise_inv_Y = torch.matmul(K_noise_inv, self.train_Y)
        
    def forward(self, X: Tensor) -> Tensor:
        
        # X: N,d
        if len(X.shape) == 3: X = X.squeeze(1)

        K_st_X = self.k(X, self.train_X).evaluate().unsqueeze(-1)
        D = (self.train_X.unsqueeze(0)-X.unsqueeze(1))
        grad_K_st_X = K_st_X * D @ self.Theta_inv

        dmu_dx = torch.linalg.matmul(grad_K_st_X.transpose(1,2),
                                     self.K_noise_inv_Y).squeeze(-1)

        grad_norm = torch.linalg.vector_norm(dmu_dx, dim=-1)
        return grad_norm.clamp_min(1e-8) # N
