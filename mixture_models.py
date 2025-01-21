import time

import torch.nn as nn
from dataset import EuropeDataset
import torch
import torch.nn.functional as F

# np.random.seed(42)
torch.manual_seed(42)

def normalize_tensor(tensor, d):
    """
    Normalize the input tensor along the specified axis to have a mean of 0 and a std of 1.
    
    Parameters:
        tensor (torch.Tensor): Input tensor to normalize.
        d (int): Axis along which to normalize.
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = torch.mean(tensor, dim=d, keepdim=True)
    std = torch.std(tensor, dim=d, keepdim=True)
    normalized = (tensor - mean) / std
    return normalized


def loss_function_umm(log_likelihood):
    """
    Compute the negative log-likelihood loss.
    Args:
        log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

    Returns:
        torch.Tensor: Negative log-likelihood.
    """
    return -torch.mean(log_likelihood)


class GMM(nn.Module):
    def __init__(self, n_components):
        """
        Gaussian Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of Gaussian components.
        """
        super().__init__()
        self.n_components = n_components

        # Mixture weights (logits to be soft maxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Means of the Gaussian components (n_components x 2 for 2D data)
        self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log of the variance of the Gaussian components (n_components x 2 for 2D data)
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))  # Log-variances (diagonal covariance)


    def forward(self, X):
        """
        Compute the log-likelihood of each sample in the batch.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood for each sample, shape (n_samples,).
        """
        # Compute log p(k) (log prior)
        log_p_k = self._compute_log_prior()  # Shape: (n_components,)

        # Compute log p(x|k) (log conditionals) for all components
        log_p_x_given_k = self._compute_log_conditionals(X)  # Shape: (n_samples, n_components)

        # Add log p(k) to log p(x|k)
        log_probs = log_p_x_given_k + log_p_k  # Broadcasting log p(k) to match shape

        # Compute log p(x) = log sum exp(log p(x|k) + log p(k)) across components
        log_likelihood = torch.logsumexp(log_probs, dim=1)  # Shape: (n_samples,)

        return log_likelihood

    def _compute_log_prior(self):
        """
        Compute log p(k) (log of mixture weights).

        Returns:
            torch.Tensor: Log prior probabilities for each component, shape (n_components,).
        """
        return F.log_softmax(self.weights, dim=0)

    def _compute_log_conditionals(self, X):
        """
        Compute log p(x|k) for all samples and components.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-conditionals for each sample and component, shape (n_samples, n_components).
        """
        log_p_x_given_k = []

        for k in range(self.n_components):
            # Parameters for the k-th Gaussian component
            mean_k = self.means[k]  # Shape: (2,)
            log_var_k = self.log_variances[k]  # Shape: (2,)
            var_k = torch.exp(log_var_k)  # Convert log-variance to variance

            # Compute log p(x|k) for all samples
            component_log_prob = (
                    - torch.log(torch.tensor(2 * torch.pi))  # Constant term
                    - torch.log(var_k[0])  # log(σ_k1^2)
                    - torch.log(var_k[1])  # log(σ_k2^2)
                    - 0.5 * (
                            ((X[:, 0] - mean_k[0]) ** 2) / var_k[0]  # (x1 - μ_k1)^2 / σ_k1^2
                            + ((X[:, 1] - mean_k[1]) ** 2) / var_k[1]  # (x2 - μ_k2)^2 / σ_k2^2
                    )
            )  # Shape: (n_samples,)

            log_p_x_given_k.append(component_log_prob)

        # Stack the log-conditionals for all components
        return torch.stack(log_p_x_given_k, dim=1)  # Shape: (n_samples, n_components)

    def sample(self, n_samples):
        """
        Sample from the Gaussian Mixture Model.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples of shape (n_samples, d).
        """
        # Get mixture probabilities (p(k)) using softmax over weights (logits)
        p_k = torch.softmax(self.weights, dim=0)  # Shape: (n_components,)

        # Sample component indices (k ~ p(k)) using torch.multinomial
        component_indices = torch.multinomial(p_k, n_samples, replacement=True)  # Shape: (n_samples,)

        # Prepare samples
        samples = []
        for k in component_indices:
            # Get parameters for the sampled component
            mean_k = self.means[k]  # Shape: (2,)
            log_var_k = self.log_variances[k]  # Shape: (2,)
            std_k = torch.sqrt(torch.exp(log_var_k))  # Standard deviation: σ_k

            # Sample from the Gaussian: x ~ N(μ_k, Σ_k)
            z = torch.randn(mean_k.shape)  # Standard normal: z ~ N(0, I)
            x = mean_k + z * std_k  # Transform to the component's distribution
            samples.append(x)

        # Stack all samples into a single tensor
        return torch.stack(samples)  # Shape: (n_samples, 2)

    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific Gaussian component.

        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        # Retrieve parameters for the specified Gaussian component
        mean_k = self.means[label]  # Mean of the label-th component, shape: (2,), e.g [1, 2]
        log_var_k = self.log_variances[label]  # Log-variance of the label-th component, shape: (2,)
        std_k = torch.sqrt(torch.exp(log_var_k))  # Standard deviation: σ = sqrt(exp(log(σ^2))), shape: (2,)

        # Sample from the standard normal distribution
        z = torch.randn((n_samples, mean_k.shape[0]))  # Shape: (n_samples, 2), z ~ N(0, I)

        # Transform samples to the Gaussian's distribution: x = μ + z * σ
        x = mean_k + z * std_k  # Broadcasting is applied automatically

        return x  # Shape: (n_samples, 2)


def loss_function_gmm(log_likelihood):
    """
    Compute the negative log-likelihood loss.
    Args:
        log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

    Returns:
        torch.Tensor: Negative log-likelihood.
    """
    return -torch.mean(log_likelihood)


class UMM(nn.Module):
    def __init__(self, n_components):
        """
        Uniform Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of uniform components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be soft maxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Center value of the uniform components (n_components x 2 for 2D data)
        self.centers = nn.Parameter(torch.randn(n_components, 2))

        # Log of size of the uniform components (n_components x 2 for 2D data)
        self.log_sizes = nn.Parameter(torch.log(torch.ones(n_components, 2) + torch.rand(n_components, 2)*0.2))

    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """
        # Ensure X is of shape (n_samples, 2)
        assert X.shape[1] == 2

        # Calculate the mixture weights using softmax
        log_weights = torch.log_softmax(self.weights, dim=0)  # Shape: (n_components,)

        # Calculate the log-probability of each component
        log_probs = []
        for i in range(self.n_components):
            center = self.centers[i]  # Shape: (2,)
            size = torch.exp(self.log_sizes[i])  # Shape: (2,)
            lower_bound = center - size / 2
            upper_bound = center + size / 2
            within_bounds = torch.all((X >= lower_bound) & (X <= upper_bound), dim=1)
            uniform_log_prob = -torch.log(size.prod())  # Log-probability of uniform distribution
            log_prob = within_bounds * uniform_log_prob + (~within_bounds) * float('-inf')
            log_probs.append(log_prob + log_weights[i])

        # Stack log-probabilities and compute log-sum-exp
        log_probs = torch.stack(log_probs, dim=1)  # Shape: (n_samples, n_components)
        log_likelihood = torch.logsumexp(log_probs, dim=1)  # Shape: (n_samples,)

        return log_likelihood

    def sample(self, n_samples):
        """
        Generate samples from the UMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        # Sample component indices based on the mixture weights
        weights = torch.softmax(self.weights, dim=0)
        component_indices = torch.multinomial(weights, n_samples, replacement=True)

        # Generate samples
        samples = []
        for idx in component_indices:
            center = self.centers[idx]
            size = torch.exp(self.log_sizes[idx])
            sample = center + (torch.rand(2) - 0.5) * size
            samples.append(sample)

        return torch.stack(samples)

    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        center = self.centers[label]
        size = torch.exp(self.log_sizes[label])
        samples = center + (torch.rand(n_samples, 2) - 0.5) * size
        return samples


# Example Usage
if __name__ == "__main__":
    # log_probs = [
    #     torch.tensor([-2.0, -1.5]),  # Component 1
    #     torch.tensor([-1.8, -1.3]),  # Component 2
    #     torch.tensor([-2.5, -1.7]) ]  # Component 3
    # log_probs = torch.stack(log_probs, dim=1)  # Shape: (n_samples, n_components)
    # print(log_probs.shape)
    t = torch.tensor([1, 2])
    print(t.shape)

    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    num_labels = train_dataset.get_number_of_labels()

    batch_size = 4096
    num_epochs = 50
    # Use Adam optimizer

    #TODO: Determine learning rate
    # learning_rate for GMM = 0.01
    # learning_rate for UMM = 0.001
    
    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #### YOUR CODE GOES HERE ####

    # Initialize the UMM model
    n_components = [1, 5, 10, num_labels]

    # Set the learning rate
    learning_rate = 0.001

    # Use the Adam optimizer

    # Training loop
    loss = None
    for components in n_components:
        total_loss = 0.0
        print(f"------------Number of components: {components}-----------------")
        gmm = GMM(components)
        optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            number_of_batches = 0
            batch_time = None
            for batch in train_loader:
                start_time = time.time()
                features, _ = batch  # Assuming labels are not needed
                epoch_log_likelihood = gmm(features)
                loss = loss_function_gmm(epoch_log_likelihood)
                optimizer.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
                number_of_batches += 1
                end_time = time.time()
                batch_time = end_time - start_time
            epoch_loss /= number_of_batches
            total_loss += epoch_loss
            print(f"Epoch {epoch + 1}/{num_epochs}, Time: {batch_time:.3f}s, "
                  f"Loss: {loss.item():.4f}")
        total_loss /= num_epochs
        print(f"Loss for {components} components: {total_loss}")


