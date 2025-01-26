import math
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import DataHandler
import matplotlib.pyplot as plt

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    @staticmethod
    def forward(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        #### YOUR CODE HERE ####
        # TIP:
        # It is common proactive to initialize a single Linear layer to map each token to its query, key, and value, i.e. nn.Linear(self.n_embd, 3 * self.n_embd)
        # After applying the linear layer on a token embedding you can split the layer's output to key, query, and value
        # The output key/query/value is of dimension n_embd, in practice this includes the embeddings for all heads,
        # therefore, embedding = [embd_1, embd_2, ... embd_n heads]. You can rearrange as you please in the forward pass.

        # Linear layer to compute query, key, and value in a single operation
        self.qkv = nn.Linear(n_embd, 3 * n_embd)

        # Linear layer to project concatenated output back to n_embd
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask (upper triangular matrix with -inf above the diagonal)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                      .view(1, 1, block_size, block_size))


    def forward(self, x):
        #### YOUR CODE HERE ####
        # Compute queries, keys, and values. Expected shape [batch_size, n_heads, sequence_length n_embd/n_head]

        # Compute normalized attention matrix (Q@K.T)/sqrt(d_k), Expected shape [batch_size, n_heads, sequence_length, sequence_length]
        # NOTE: the dimension d_k refers to the embedding dimension of the keys which is n_embd/num_heads

        # Mask, this is casual self-attention, you need to mask the score of each token with the tokens that come after it in the sequence
        # Fill all values above the diagonal with -float('inf'), this ensures these entries will be zeroed after softmax

        # Apply softmax on each row of the masked normalized attention matrix and perform matrix multiplication with the values
        # Expected shape [batch_size, n_heads, sequence_length, n_embd/n_head]

        # Re-Assemble all head outputs side by side. Expected shape [batch_side, sequence_length, n_embd]

        # output projection
        B, T, C = x.size()  # shapes of X
        qkv = self.qkv(x)  # Shape: [B, T, 3 * n_embd]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Split into q, k, v, each with shape: [B, T, n_embd]

        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # Shape: [B, n_h, T, d_k]
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # Shape: [B, n_h, T, d_k]
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # Shape: [B, n_h, T, d_k]

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(k.size(-1))  # Shape: [B, n_h, T, T]
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, -float('inf'))  # Shape: [B, n_h, T, T]

        attention = torch.softmax(scores, dim=-1)  # Shape: [B, n_h, T, T]
        attention = attention @ v # Shape: [B, n_h, T, d_k]
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)  # Shape: [B, T, n_embd]
        attention = self.proj(attention)  # Shape: [B, T, n_embd]

        return attention


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, self.n_embd),
            wpe=nn.Embedding(block_size, self.n_embd),
            h=nn.ModuleList([Block(n_head, n_embd, block_size) for _ in range(self.n_layer)]),
            ln_f=nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits


def train_model(
        train_path,
        test_path=None,
        model=None,
        block_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10,
        losses_accuracies_path="losses_accuracies.pt"

):
    data_handler = DataHandler(train_path, test_path, block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = data_handler.get_vocab_size()
    if model is None:
        model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    print('Using device:', device)

    train_set = data_handler.get_dataset('train')
    test_set = data_handler.get_dataset('test')

    # set up the dataloader
    train_loader = DataLoader(
        train_set,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=int(1e5)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
    )
    test_loader = None
    if test_set:
        test_loader = DataLoader(
            test_set,
            sampler=torch.utils.data.RandomSampler(test_set, replacement=False, num_samples=int(1e4)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,
        )

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for ep in range(epochs):
        model.train()
        train_total_loss = 0
        train_correct_predictions = 0
        train_total_predictions = 0

        for i, batch in enumerate(tqdm(train_loader)):
            #### YOUR CODE HERE ####
            inputs, targets = batch
            inputs.long().to(device), targets.long().to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()

            last_char_logits = logits[:, -1, :]
            predicted = torch.argmax(last_char_logits, dim=1)
            train_correct_predictions += (predicted == targets[:, -1]).sum().item()
            train_total_predictions += targets.size(0)

        train_average_loss = train_total_loss / len(train_loader)
        train_average_accuracy = train_correct_predictions / train_total_predictions
        train_losses.append(train_average_loss)
        train_accuracies.append(train_average_accuracy)

        print(f"\nEpoch {ep + 1}: Train Accuracy: {train_average_accuracy:.4f}, Train Loss: {train_average_loss:.4f}")
        model.eval()
        with torch.no_grad():
            test_total_loss = 0
            test_correct_predictions = 0
            test_total_predictions = 0
            for i, batch in enumerate(tqdm(test_loader)):
                inputs, targets = batch
                inputs.long().to(device), targets.long().to(device)

                # Forward pass
                logits = model(inputs)

                # Compute loss
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                test_total_loss += loss.item()

                last_char_logits = logits[:, -1, :]
                predicted = torch.argmax(last_char_logits, dim=1)
                test_correct_predictions += (predicted == targets[:, -1]).sum().item()
                test_total_predictions += targets.size(0)

            test_average_loss = test_total_loss / len(test_loader)
            test_average_accuracy = test_correct_predictions / test_total_predictions
            test_losses.append(test_average_loss)
            test_accuracies.append(test_average_accuracy)

            print(f"\nEpoch {ep + 1}: Test Accuracy: {test_average_accuracy:.4f}, Test Loss: {test_average_loss:.4f}")


            # Complete the sentence:
            print("\nGenerating sentences without top-5 sampling:")
            sentence="the "
            for i in range(3):
                new_sentence = sentence
                for _ in range(30):
                        tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:])).unsqueeze(0).to(device)
                        logits = model(tokens)
                        probs = torch.softmax(logits[0, -1, :], dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                        new_sentence += data_handler.decoder([next_token])
                print(f'Generated sentence {i + 1}:', new_sentence)

            # Complete the sentence using top-k sampling
            print("\nGenerating sentences with top-5 sampling:")
            top_k = 5  # Consider the top 5 tokens
            # sentence = "the "
            for i in range(3):  # Generate 3 sentences
                new_sentence = sentence
                for _ in range(30):  # Generate 20 tokens per sentence
                    tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:])).unsqueeze(0).to(device)
                    logits = model(tokens)
                    probs = torch.softmax(logits[0, -1, :], dim=-1)

                    top_probs, top_indices = torch.topk(probs, top_k)  # Top-k probabilities and indices
                    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities
                    next_token = top_indices[torch.multinomial(top_probs, 1)].item()  # Sample from the top-k probabilities
                    new_sentence += data_handler.decoder([next_token])  # Append the sampled token
                print(f'Generated sentence {i + 1}:', new_sentence)

    # Save the model after training
    # Save the model and metrics
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }, losses_accuracies_path)
    print(f"Model and metrics saved to {losses_accuracies_path}")
    return train_losses, train_accuracies, test_losses, test_accuracies


def plot_losses_accuracies(saved_data_path="losses_accuracies.pt"):
    # Load saved data
    saved_data = torch.load(saved_data_path)
    train_losses = saved_data['train_losses']
    train_accuracies = saved_data['train_accuracies']
    test_losses = saved_data['test_losses']
    test_accuracies = saved_data['test_accuracies']

    # Plotting
    fig, (loss, accuracy) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)

    # Accuracy plot
    accuracy.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    accuracy.plot(epochs, test_accuracies, 'r-', label='Testing Accuracy')
    accuracy.set_title('Accuracies')
    accuracy.set_xlabel('Epoch')
    accuracy.set_ylabel('Accuracy')
    accuracy.legend()
    # Loss plot
    loss.plot(epochs, train_losses, 'b-', label='Training Loss')
    loss.plot(epochs, test_losses, 'r-', label='Testing Loss')
    loss.set_title('Losses')
    loss.set_xlabel('Epoch')
    loss.set_ylabel('Loss')
    loss.legend()



    plt.tight_layout()
    plt.show()

    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Testing Loss: {test_losses[-1]:.4f}")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final Testing Accuracy: {test_accuracies[-1]:.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    train_model('train_shakespeare.txt', 'test_shakespeare.txt')
    plot_losses_accuracies()


