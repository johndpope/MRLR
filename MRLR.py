import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
'''
Multi-resolution approach: The MRLR method is like looking at a picture at different zoom levels. First, you capture the broad strokes (lower resolution), then you focus on finer details (higher resolution).
PARAFAC decomposition: This is similar to breaking down a complex shape into simpler components. For a 3D tensor, imagine decomposing a complex 3D object into a set of 1D "building blocks" that, when combined, approximate the original object.
Residual updates: This process is like an artist refining a sketch. First, they draw the main outlines, then they focus on the details that are missing from the initial sketch.
Khatri-Rao product: This can be thought of as a way to combine information from different dimensions. It's like creating a "super factor" that represents the interaction between all other factors.
Convergence check: This is similar to a painter stepping back from their canvas to see if the painting looks "close enough" to the subject. If it does, they stop; if not, they continue refining.
'''
class MRLR:
    def __init__(self, tensor, partitions, ranks):
        """
        Initialize the MRLR decomposition.
        
        Args:
        tensor (torch.Tensor): The input tensor to be decomposed.
        partitions (list of list of list): The partitions for multi-resolution decomposition.
        ranks (list of int): The ranks for each partition.
        """

        self.tensor = tensor
        self.partitions = partitions
        self.ranks = ranks
        print(f"Tensor shape: {self.tensor.shape}")
        self._validate_partitions()
        self.factors = self._initialize_factors()

    def _validate_partitions(self):
        """Validate that partitions match tensor dimensions."""
        tensor_modes = set(range(self.tensor.dim()))
        for i, partition in enumerate(self.partitions):
            partition_modes = set(mode for group in partition for mode in group)
            print(f"Partition {i}: {partition}")
            print(f"Tensor modes: {tensor_modes}")
            print(f"Partition modes: {partition_modes}")
            if partition_modes != tensor_modes:
                print(f"Warning: Partition {i} does not match tensor dimensions.")
                
    def _initialize_factors(self):
        """Initialize factors for each partition."""
        factors = []
        for partition, rank in zip(self.partitions, self.ranks):
            partition_factors = []
            for mode_group in partition:
                size = 1
                for mode in mode_group:
                    size *= self.tensor.shape[mode]
                partition_factors.append(torch.randn(size, rank))
            factors.append(partition_factors)
        return factors

    def _unfold(self, tensor, partition):
        """Unfold the tensor according to the given partition."""
        modes = [mode for group in partition for mode in group]
        return tensor.permute(*modes).reshape(tensor.shape[modes[0]], -1)

    def _fold(self, unfolded, partition, original_shape):
        """Fold the unfolded tensor back to its original shape."""
        intermediate_shape = []
        for mode_group in partition:
            for mode in mode_group:
                intermediate_shape.append(original_shape[mode])
        return unfolded.reshape(intermediate_shape)

    def _parafac(self, tensor, rank, max_iter=100, tol=1e-4):
        """Perform PARAFAC decomposition."""
        # Initialize factors randomly
        # Analogy: Think of this as creating a rough sketch for each dimension of the tensor
        factors = [torch.randn(s, rank) for s in tensor.shape]
        
        for _ in range(max_iter):
            old_factors = [f.clone() for f in factors]
            for mode in range(len(factors)):
                # Unfold the tensor for the current mode
                # Analogy: This is like "flattening" the tensor to focus on one dimension at a time
                unfold_mode = self._unfold(tensor, [[mode], list(range(mode)) + list(range(mode+1, tensor.dim()))])
                
                # Compute Khatri-Rao product
                # Analogy: This is like creating a "super factor" that combines all other factors
                khatri_rao_prod = factors[(mode+1) % len(factors)]
                for i in range(2, len(factors)):
                    current_factor = factors[(mode+i) % len(factors)]
                    khatri_rao_prod = torch.einsum('ir,jr->ijr', khatri_rao_prod, current_factor).reshape(-1, rank)
                
                # Update the current factor
                # Analogy: This is like adjusting our "sketch" to better match the original tensor
                V = khatri_rao_prod.t() @ khatri_rao_prod
                factor_update = unfold_mode @ khatri_rao_prod
                
                try:
                    factors[mode] = factor_update @ torch.pinverse(V)
                    # Handle potential numerical instabilities
                    factors[mode] = torch.nan_to_num(factors[mode], nan=0.0, posinf=1e10, neginf=-1e10)
                except RuntimeError as e:
                    print(f"Error in PARAFAC: {e}")
                    print(f"V shape: {V.shape}")
                    print(f"factor_update shape: {factor_update.shape}")
                    return factors  # Return the current factors and exit
            
            # Check for convergence
            # Analogy: This is like checking if our "sketch" is close enough to the original
            if all(torch.norm(f - old_f) < tol for f, old_f in zip(factors, old_factors)):
                break
        
        return factors

    def decompose(self, max_iter=100, tol=1e-4):
        """Perform MRLR decomposition."""
        residual = self.tensor.clone()
        approximations = []

        for partition, rank in zip(self.partitions, self.ranks):
            # Unfold the residual tensor according to the current partition
            unfolded = self._unfold(residual, partition)
            
            # Perform PARAFAC decomposition on the unfolded residual
            factors = self._parafac(unfolded, rank, max_iter, tol)
            
            # Reconstruct the approximation from the factors
            if len(factors) == 2:
                approximation = self._fold(factors[0] @ factors[1].T, partition, residual.shape)
            else:
                # For higher-order tensors, we need a more general reconstruction method
                approximation = self._fold(self._reconstruct_from_factors(factors), partition, residual.shape)
            
            approximations.append(approximation)
            residual -= approximation

        return approximations

    def _reconstruct_from_factors(self, factors):
        """Reconstruct the tensor from its factors."""
        reconstructed = factors[0]
        for factor in factors[1:]:
            reconstructed = torch.einsum('...i,ji->...j', reconstructed, factor)
        return reconstructed

    
    
    def reconstruct(self):
        """Reconstruct the tensor from its decomposition."""
        # Sum up all the approximations to get the final reconstruction
        # Analogy: This is like combining all our "sketches" to recreate the full "picture"
        return sum(self.decompose())
    


    def visualize_decomposition(self, approximations, residuals):
        n_rows = len(approximations) + 1
        fig = plt.figure(figsize=(20, 5*n_rows))
        
        # Plot original tensor
        self._plot_tensor(fig, n_rows, 1, self.tensor, "Original Tensor")
        
        # Plot approximations and residuals
        for i, (approx, res) in enumerate(zip(approximations, residuals[1:]), start=1):
            self._plot_tensor(fig, n_rows, i*2, approx, f"Approximation {i}")
            self._plot_tensor(fig, n_rows, i*2+1, res, f"Residual {i}")
        
        # Plot final reconstruction
        reconstruction = sum(approximations)
        self._plot_tensor(fig, n_rows, n_rows*2, reconstruction, "Final Reconstruction")
        
        plt.tight_layout()
        plt.show()

    def _plot_tensor(self, fig, n_rows, pos, tensor, title):
        ax = fig.add_subplot(n_rows, 2, pos, projection='3d')
        
        if tensor.dim() == 2:
            X, Y = np.meshgrid(range(tensor.shape[1]), range(tensor.shape[0]))
            Z = tensor.numpy()
        elif tensor.dim() == 3:
            X, Y = np.meshgrid(range(tensor.shape[1]), range(tensor.shape[0]))
            Z = tensor[:,:,0].numpy()
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# ... (rest of the code remains the same)

# ... (rest of the code remains the same)

def normalized_frobenius_error(original, approximation):
    """Compute the Normalized Frobenius Error."""
    return torch.norm(original - approximation) / torch.norm(original)



if __name__ == "__main__":
    # Create a sample 24x27x23 tensor
    tensor = torch.randn(24, 27, 23)
    
    # Define partitions for multi-resolution decomposition
    partitions = [
        [[0], [1, 2]],  # Matrix unfolding
        [[0], [1], [2]]  # Full tensor
    ]
    
    # Define ranks for each partition
    ranks = [10, 5]
    
    # Create MRLR object and perform decomposition
    mrlr = MRLR(tensor, partitions, ranks)
    try:
        approximations, residuals = mrlr.decompose()
        
        # Visualize the decomposition
        mrlr.visualize_decomposition(approximations, residuals)
        
        # Reconstruct the tensor
        reconstructed = sum(approximations)
        
        # Compute error
        error = normalized_frobenius_error(tensor, reconstructed)
        print(f"Normalized Frobenius Error: {error.item()}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()