from tqdm import tqdm
import pandas as pd
import pickle


from torch_geometric.loader import DataLoader
from data import ProteinECNumGraphDataset
from egnn_model import GraphEC
import torch.nn as nn
import torch

device = "cuda:0"
config = {
    'node_input_dim': 9+184,  # 9 + 184, # precomputed + updated
    'edge_input_dim': 450,
    'hidden_dim': 1024,
    'layer': 1,
    'augment_eps': 0.15,
    'class_align_weight': 0.2,
    'batch_size': 4,
    'folds': 5,
    'r': 16,
    'num_workers': 8,
    "random_seed": 0
}

torch.manual_seed(config["random_seed"])


def get_ec_map():
    with open("./data_index/ec_map.pkl", "rb") as f:
        ec_map = pickle.load(f)
    return ec_map


def get_data():
    print("Loading Training Set")
    with open("./data_index/training_set.pkl", "rb") as f:
        train_data = pickle.load(f)

    train_dataset = ProteinECNumGraphDataset(
        train_data, radius=config['r'], split="training")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=config['num_workers'],
        prefetch_factor=2
    )

    print("Loading Validation Set")
    val_dataset = ProteinECNumGraphDataset(
        train_data, radius=config['r'], split="validation")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_workers'],
        prefetch_factor=2
    )
    return train_dataloader, val_dataloader


def get_last_loss(pred, true, all_ec_map):
    device = pred.device
    bce_loss = [
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7]).to(device)),
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor([74]).to(device)),
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor([258]).to(device)),
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5069]).to(device)),
    ]
    loss = bce_loss[-1](pred.view(-1), true[-1].view(-1))
    for i, ec_map in reversed([i for i in enumerate(all_ec_map)]):
        ec_map = [(
            sum(ec_map[:i]), sum(ec_map[:i+1])
        )for i in range(len(ec_map))]
        pred = torch.stack([
            pred[:, p1:p2].max(axis=1).values for p1, p2 in ec_map]).T
        # sum([4/10, 3/10, 2/10, 1/10]) = 1
        loss += bce_loss[i](pred.reshape(-1), true[i].view(-1))*(i+1)/10
    return loss


def train_model(model, optimizer, ec_map, train_dataloader, val_dataloader, epochs):
    all_train_loss = []
    all_val_loss = []
    model.train()
    for epoch in range(epochs):
        model, loss = train_step(
            model, optimizer, ec_map, train_dataloader, epoch, epochs)
        print(f"Epoch {epoch+1}, Loss: {loss}")
        all_train_loss.append(loss / len(train_dataloader))
        val_loss = validate_step(model, ec_map, val_dataloader)
        print("Val Loss:", val_loss)
        all_val_loss.append(val_loss)
    print("Training complete.")
    return model, all_train_loss, all_val_loss


def train_step(model, optimizer, ec_map, train_dataloader, epoch, epochs):
    running_loss = 0.0
    tqdm_bar = tqdm(train_dataloader,
                    desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
    for data in tqdm_bar:
        data = data.to(device)
        output = model.forward(
            data.X, data.structure_feat, data.seq_feat,
            data.edge_index, data.batch
        )
        true = (data.ec1_label, data.ec2_label, data.ec3_label, data.ec4_label)
        loss = get_last_loss(output, true, ec_map)
        tqdm_bar.set_postfix_str(f"Current Loss:{loss.item():.4f}")
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    del data, output, loss
    return model, running_loss/len(train_dataloader)


def validate_step(model, ec_map, val_dataloader):
    with torch.no_grad():  # Validate
        running_loss = 0.0
        tqdm_bar = tqdm(val_dataloader)
        for data in tqdm_bar:
            data = data.to(device)
            output = model.forward(
                data.X, data.structure_feat, data.seq_feat,
                data.edge_index, data.batch
            )
            true = (data.ec1_label, data.ec2_label,
                    data.ec3_label, data.ec4_label)
            loss = get_last_loss(output, true, ec_map)
            tqdm_bar.set_postfix_str(f"Current Loss:{loss.item():.4f}")
            running_loss += loss.item()
    del data, output, loss
    return running_loss / len(val_dataloader)


if __name__ == "__main__":
    train_dataloader, val_dataloader = get_data()
    ec_map = get_ec_map()
    train_epochs = 100
    model = GraphEC(
        config['node_input_dim'],
        config['edge_input_dim'],
        config['hidden_dim'],
        config['layer'],
        config['augment_eps'],
        device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    model, train_loss, val_loss = train_model(
        model, optimizer, ec_map,
        train_dataloader, val_dataloader, train_epochs
    )
    log = pd.DataFrame([train_loss, val_loss], index=["train", "val"]).T
    log.to_csv("test_model.csv")
    torch.save(model, "test_model.pt")
