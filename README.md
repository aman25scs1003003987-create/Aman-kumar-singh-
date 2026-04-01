import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.fc_layers(x)


# Example usage
if __name__ == "__main__":
    num_users = 1000
    num_items = 500

    model = NCF(num_users, num_items)

    user = torch.randint(0, num_users, (10,))
    item = torch.randint(0, num_items, (10,))

    output = model(user, item)
    print("NCF Output:\n", output)
