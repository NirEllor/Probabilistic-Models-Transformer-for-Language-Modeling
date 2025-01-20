
import torch
import torch.nn as nn
from dataset import EuropeDataset

np.random.seed(42)
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


class GMM(nn.Module):
    def __init__(self, n_components):
        """
        Gaussian Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of Gaussian components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Means of the Gaussian components (n_components x 2 for 2D data)
        self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log of the variance of the Gaussian components (n_components x 2 for 2D data)
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))  # Log-variances (diagonal covariance)

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
            mean = self.means[i]  # Shape: (2,)
            var = torch.exp(self.log_variances[i])  # Shape: (2,)
            cov = torch.diag(var)  # Shape: (2, 2)
            dist = torch.distributions.MultivariateNormal(mean, cov)
            log_prob = dist.log_prob(X)  # Shape: (n_samples,)
            log_probs.append(log_prob + log_weights[i])

        # Stack log-probabilities and compute log-sum-exp
        log_probs = torch.stack(log_probs, dim=1)  # Shape: (n_samples, n_components)
        log_likelihood = torch.logsumexp(log_probs, dim=1)  # Shape: (n_samples,)

        return log_likelihood

    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        return -torch.mean(log_likelihood)


    def sample(self, n_samples):
        """
        Generate samples from the GMM model.
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
            mean = self.means[idx]
            var = torch.exp(self.log_variances[idx])
            cov = torch.diag(var)
            dist = torch.distributions.MultivariateNormal(mean, cov)
            sample = dist.sample()
            samples.append(sample)

        return torch.stack(samples)

    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific Gaussian component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        mean = self.means[label]
        var = torch.exp(self.log_variances[label])
        cov = torch.diag(var)
        dist = torch.distributions.MultivariateNormal(mean, cov)
        samples = dist.sample((n_samples,))
        return samples


class UMM(nn.Module):
    def __init__(self, n_components):
        """
        Uniform Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of uniform components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
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

    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        return -torch.mean(log_likelihood)

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
    
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

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
    n_components = 3  # Set the number of components as needed
    umm = UMM(n_components)

    # Set the learning rate
    learning_rate = 0.001

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(umm.parameters(), lr=learning_rate)

    # Training loop
    loss = None
    for epoch in range(num_epochs):
        for batch in train_loader:
            features, _ = batch  # Assuming labels are not needed
            log_likelihood = umm(features)
            loss = umm.loss_function(log_likelihood)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")



