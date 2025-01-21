# import time

import torch.nn as nn
import dataset
from dataset import EuropeDataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_FEATURES = 2
EPOCHS = [1, 10, 20, 30, 40, 50]


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


def plot_gmm_samples(gmm_model, n_samples=1000, n_components=1, is_num_labels=False, epoch_number=1):
    """
    Display a scatter plot with samples from the given GMM.

    Args:
        gmm (GMM): The Gaussian Mixture Model object.
        n_samples (int): Number of samples to generate for the scatter plot.
        :param epoch_number: epoch number
        :param is_num_labels: if the number of labels is the number of components
        :param gmm_model: The model.
        :param n_samples: Number of samples to generate.
        :param n_components: number of components
    """
    # Generate samples from the GMM using the sample function
    samples = gmm_model.sample(n_samples)  # Shape: (n_samples, 2)

    # Convert samples to numpy for plotting
    samples = samples.detach().numpy()

    # Plot the samples
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, edgecolors='k')
    if is_num_labels:
        plt.title(f"Epoch {epoch_number}: Scatter Plot of {n_samples} Samples from {n_components} components GMM")
    else:
        plt.title(f"Scatter Plot of {n_samples} Samples from {n_components} components GMM")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


def plot_loss_from_file(file_path, n_components):
    """
    Load the loss tensor from a file and plot the train and test losses.

    Args:
        file_path (str): Path to the file containing the saved loss tensor.
        n_components (list): List of component counts corresponding to the rows of the loss tensor.
    """
    try:
        # Load the saved loss tensor
        loss_train_test_loaded = torch.load(file_path, weights_only=True)

        # Extract train and test losses
        train_loss = loss_train_test_loaded[:, 0].tolist()  # First column: Train losses
        test_loss = loss_train_test_loaded[:, 1].tolist()  # Second column: Test losses

        # Plot train and test losses
        plt.figure(figsize=(8, 6))
        plt.plot(n_components, train_loss, label='Train Loss', marker='o')
        plt.plot(n_components, test_loss, label='Test Loss', marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss vs. Number of Components')
        # plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        print("Ensure the file path is correct and the tensor is properly formatted.")


def plot_conditional_samples(gmm_model, n_samples_per_component=100, num_components=1, is_num_labels=False, epoch_number=1):
    """
    Display a scatter plot with samples from each Gaussian component, colored by component.

    Args:
        gmm_model (GMM): The Gaussian Mixture Model object.
        n_samples_per_component (int): Number of samples to generate per Gaussian component.
        :param epoch_number: epoch number
        :param is_num_labels: if the number of labels is the number of components
        :param n_samples_per_component: number of samples per component
        :param gmm_model: The model
        :param num_components: Number of components
    """
    all_samples = []
    all_labels = []

    # Loop through each Gaussian component
    for component in range(gmm_model.n_components):
        # Sample from the specific Gaussian component
        samples = gmm_model.conditional_sample(n_samples_per_component, component)

        # Append samples and labels
        all_samples.append(samples)
        all_labels.extend([component] * n_samples_per_component)  # Use component index as label

    # Combine all samples into one tensor
    all_samples = torch.cat(all_samples, dim=0)  # Shape: (n_samples_per_component * n_components, 2)
    all_labels = np.array(all_labels)  # Convert labels to numpy array

    # Convert samples to numpy for plotting
    all_samples = all_samples.detach().numpy()

    # Scatter plot with color-coding by component
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_samples[:, 0], all_samples[:, 1], c=all_labels, cmap='tab10', alpha=0.8, edgecolors='k')
    # plt.colorbar(scatter, label="Gaussian Component")
    if is_num_labels:
        plt.title(f"Epoch {epoch_number}: {n_samples_per_component} Samples from Each of Gaussian Component, {num_components} in total")
    else:
        plt.title(f"{n_samples_per_component} Samples from Each of Gaussian Component, {num_components} in total")

    plt.xlabel("Long")
    plt.ylabel("Lat")
    plt.grid(True)
    plt.show()


def plot_log_likelihood(_train_log_likelihood, _test_log_likelihood, _num_epochs):
    """
    Plot the training and testing mean log likelihood vs. epoch.

    Args:
        _train_log_likelihood (list): List of mean log likelihoods for each epoch during training.
        _test_log_likelihood (list): List of mean log likelihoods for each epoch during testing.
        _num_epochs (int): Total number of epochs.
    """
    epochs = list(range(1, _num_epochs + 1))  # Epoch numbers

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, _train_log_likelihood, label="Training Log-Likelihood", marker='o')
    plt.plot(epochs, _test_log_likelihood, label="Testing Log-Likelihood", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Log-Likelihood")
    plt.title("Training and Testing Mean Log-Likelihood vs. Epoch")
    # plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage
if __name__ == "__main__":
    # log_probs = [
    #     torch.tensor([-2.0, -1.5]),  # Component 1
    #     torch.tensor([-1.8, -1.3]),  # Component 2
    #     torch.tensor([-2.5, -1.7]) ]  # Component 3
    # log_probs = torch.stack(log_probs, dim=1)  # Shape: (n_samples, n_components)
    # print(log_probs.shape)
    # t = torch.tensor([1, 2])
    # print(t.shape)

    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    num_labels = train_dataset.get_number_of_labels()

    batch_size = 4096
    num_epochs = 50
    # Use Adam optimizer

    #TODO: Determine learning rate
    # learning_rate for GMM = 0.01
    # learning_rate for UMM = 0.001
    
    train_dataset.features = dataset.normalize_tensor(train_dataset.features, d=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset.features = dataset.normalize_tensor(test_dataset.features, d=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #### YOUR CODE GOES HERE ####

    number_components = [1, 5, 10, num_labels]

    # Set the learning rate
    learning_rate_gmm = 0.01
    learning_rate_umm = 0.001

    # Compute mean location for each class
    country_means = []
    for _label in range(num_labels):
        class_features = train_dataset.features[train_dataset.labels == _label]
        class_mean = torch.mean(class_features, dim=0)
        country_means.append(class_mean)

    country_means = torch.stack(country_means)  # Shape: (num_labels, 2)
    loss_train_test = torch.zeros((len(number_components), NUMBER_OF_FEATURES))

    for initialize_with_means in [False, True]:  # Train twice: random and country means
        print(f"Training GMM with {'country means' if initialize_with_means else 'random initialization'}")

        for index,components in enumerate(number_components):

            train_total_loss = 0.0
            test_total_loss = 0.0

            train_log_likelihood = []  # Reset log-likelihood tracker for training
            test_log_likelihood = []  # Reset log-likelihood tracker for testing

            print(f"------------Number of components: {components}-----------------")

            gmm = GMM(components)

            # Initialize means if specified
            if initialize_with_means and components == num_labels:
                with torch.no_grad():
                    gmm.means.copy_(country_means)

            optimizer = torch.optim.Adam(gmm.parameters(), lr=learning_rate_gmm)
            for epoch in range(num_epochs):
                print(f"--------------Epoch {epoch + 1}-----------------")

                train_epoch_loss = 0.0
                test_epoch_loss = 0.0

                train_number_of_batches = 0
                test_number_of_batches = 0

                train_log_likelihood_batches = []  # For tracking batch log-likelihoods
                test_log_likelihood_batches = []  # For tracking batch log-likelihoods

                # batch_time = None
                # Training loop
                for batch in train_loader:
                    # start_time = time.time()
                    features, _ = batch  # Assuming labels are not needed

                    epoch_log_likelihood = gmm(features)
                    train_log_likelihood_batches.append(epoch_log_likelihood)  # Track batch log-likelihood
                    loss = loss_function_gmm(epoch_log_likelihood)
                    optimizer.zero_grad()
                    loss.backward()

                    train_epoch_loss += loss.item()
                    optimizer.step()
                    train_number_of_batches += 1
                    # end_time = time.time()
                    # batch_time = end_time - start_time

                # Compute mean log-likelihood for the epoch
                train_epoch_log_likelihood_mean = torch.mean(torch.cat(train_log_likelihood_batches)).item()
                train_log_likelihood.append(train_epoch_log_likelihood_mean)

                train_epoch_loss /= train_number_of_batches
                train_total_loss += train_epoch_loss

                print(f"Train Loss: {train_epoch_loss:.4f}")
                with torch.no_grad():
                    for batch in test_loader:
                        # start_time = time.time()
                        features, _ = batch

                        epoch_log_likelihood = gmm(features)
                        test_log_likelihood_batches.append(epoch_log_likelihood)  # Track batch log-likelihood
                        loss = loss_function_gmm(epoch_log_likelihood)

                        test_epoch_loss += loss.item()
                        test_number_of_batches += 1

                        # end_time = time.time()
                        # batch_time = end_time - start_time

                    # Compute mean log-likelihood for the epoch
                    test_epoch_log_likelihood_mean = torch.mean(torch.cat(test_log_likelihood_batches)).item()
                    test_log_likelihood.append(test_epoch_log_likelihood_mean)

                    test_epoch_loss /= test_number_of_batches
                    test_total_loss += test_epoch_loss

                    print(f"Test Loss: {test_epoch_loss:.4f}")
                if (epoch + 1) in EPOCHS and components == num_labels:
                    plot_gmm_samples(gmm, 1000, num_labels,True, epoch + 1)
                    plot_conditional_samples(gmm, 100, num_labels, True, epoch + 1)
                    if epoch + 1 == num_epochs:
                        plot_log_likelihood(train_log_likelihood, test_log_likelihood, epoch + 1)

            # Average the losses across epochs
            train_total_loss /= num_epochs
            test_total_loss /= num_epochs
            print(f"Train Loss for {components} components: {train_total_loss:.4f}")
            print(f"Test Loss for {components} components: {test_total_loss:.4f}")

            # Store results using idx for indexing
            loss_train_test[index][0] = train_total_loss
            loss_train_test[index][1] = test_total_loss


            torch.save(gmm.state_dict(), 'gmm_model.pt')

            plot_gmm_samples(gmm, 1000, components)
            plot_conditional_samples(gmm, 100, components)

        loss_train_test_transposed = loss_train_test.transpose(0, 1)
        print(loss_train_test)
        torch.save(loss_train_test, 'loss_train_test.pt')
        plot_loss_from_file('loss_train_test.pt', [1, 5, 10, 'num_labels'])





