"""
Physics-Informed Neural Network (PINN) for Digital Machining Chatter Detection.
Integrates domain physics (Altintaş-Budak regenerative stability boundary)
into the neural network loss function to enforce physically consistent boundaries.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any


class MachiningPINN:
    """
    Physics-Informed Neural Network for Digital Machining Stability Modeling.
    Provides standard PyTorch architecture with physics-regularized loss.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        lambda_physics: float = 0.25,
        learning_rate: float = 0.003,
        random_state: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lambda_physics = lambda_physics
        self.lr = learning_rate
        self.random_state = random_state
        self.model = None

    def build_torch_model(self):
        """Build PyTorch neural network modules."""
        try:
            import torch
            import torch.nn as nn
            
            torch.manual_seed(self.random_state)
            
            layers = []
            prev_dim = self.input_dim
            for h in self.hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.15))
                prev_dim = h
                
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            
            self.model = nn.Sequential(*layers)
            return self.model
        except ImportError:
            raise ImportError("PyTorch is required for MachiningPINN.")

    @staticmethod
    def physics_regularization_loss(
        y_prob,
        axial_depth_m,
        boundary_m,
        lambda_phys: float = 0.25
    ):
        """
        Physics constraint loss enforcing Altintaş-Budak stability lobes:
        Penalizes predictions where:
        1. Model predicts Stable (prob close to 0) when depth b > blim(omega).
        2. Model predicts Chatter (prob close to 1) when depth b < blim(omega).
        """
        import torch
        # Distance to boundary: positive = above boundary (Chatter zone), negative = below boundary (Stable zone)
        delta_depth = axial_depth_m - boundary_m
        
        # Violation 1: In chatter zone (delta > 0), but predicted stable (y_prob -> 0)
        violation_chatter = torch.clamp(delta_depth, min=0.0) * (1.0 - y_prob)
        
        # Violation 2: In stable zone (delta < 0), but predicted chatter (y_prob -> 1)
        violation_stable = torch.clamp(-delta_depth, min=0.0) * y_prob
        
        phys_loss = torch.mean(violation_chatter + violation_stable)
        return lambda_phys * phys_loss

    def train_pinn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        depth_train: np.ndarray,
        boundary_train: np.ndarray,
        epochs: int = 150,
        batch_size: int = 64
    ) -> Dict[str, list]:
        """Train PINN model with combined BCE and physics loss."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader

        self.build_torch_model()
        
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        d_t = torch.tensor(depth_train, dtype=torch.float32).unsqueeze(1)
        b_t = torch.tensor(boundary_train, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_t, y_t, d_t, b_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        bce_loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        history = {"bce_loss": [], "phys_loss": [], "total_loss": []}
        
        self.model.train()
        for epoch in range(epochs):
            ep_bce, ep_phys, ep_total = 0.0, 0.0, 0.0
            batches = 0
            for batch_x, batch_y, batch_d, batch_b in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                
                loss_bce = bce_loss_fn(preds, batch_y)
                loss_phys = self.physics_regularization_loss(preds, batch_d, batch_b, self.lambda_physics)
                total_loss = loss_bce + loss_phys
                
                total_loss.backward()
                optimizer.step()
                
                ep_bce += loss_bce.item()
                ep_phys += loss_phys.item()
                ep_total += total_loss.item()
                batches += 1
                
            history["bce_loss"].append(ep_bce / batches)
            history["phys_loss"].append(ep_phys / batches)
            history["total_loss"].append(ep_total / batches)
            
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict chatter probabilities."""
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            probs = self.model(X_t).numpy().flatten()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary chatter class."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def export_onnx(self, onnx_path: str):
        """Export PyTorch PINN model to ONNX for ultra-low latency inference."""
        import torch
        self.model.eval()
        dummy_input = torch.randn(1, self.input_dim, dtype=torch.float32)
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_features'],
                output_names=['chatter_prob'],
                dynamic_axes={'input_features': {0: 'batch_size'}, 'chatter_prob': {0: 'batch_size'}},
                dynamo=False
            )
        except Exception:
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                input_names=['input_features'],
                output_names=['chatter_prob']
            )
        print(f"[+] PINN ONNX model exported to: {onnx_path}")
