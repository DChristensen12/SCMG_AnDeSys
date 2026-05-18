import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from config.config import Config


def train_temporal_gnn(
    model,
    train_sequences,
    train_targets,
    edge_index,
    val_sequences=None,
    val_targets=None,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    device=Config.DEVICE,
):
    """
    Train temporal GNN. Uses mixed-precision acceleration on CUDA; falls back
    to plain fp32 on CPU (autocast on CPU breaks oneDNN's LSTM kernel and gives
    no speedup anyway since CPUs lack tensor cores).

    Learns baseline creek physics via reconstruction (MSE loss).
    """
    model = model.to(device)
    edge_index = edge_index.to(device)

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    use_amp = device_type == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler(device_type) if use_amp else None

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"--- Starting Training on {device_type.upper()} "
          f"({'mixed-precision' if use_amp else 'fp32'}) ---")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        # Mini-batch training
        for i in range(0, len(train_sequences), batch_size):
            batch_seq = torch.FloatTensor(train_sequences[i:i+batch_size]).to(device)
            batch_target = torch.FloatTensor(train_targets[i:i+batch_size]).to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device_type):
                    predictions = model(
                        batch_seq,
                        edge_index,
                        batch_size=len(batch_seq),
                        num_nodes=batch_seq.shape[2],
                    )
                    loss = criterion(predictions, batch_target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(
                    batch_seq,
                    edge_index,
                    batch_size=len(batch_seq),
                    num_nodes=batch_seq.shape[2],
                )
                loss = criterion(predictions, batch_target)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        if val_sequences is not None:
            model.eval()
            with torch.no_grad():
                val_seq = torch.FloatTensor(val_sequences).to(device)
                val_tgt = torch.FloatTensor(val_targets).to(device)
                if use_amp:
                    with autocast(device_type=device_type):
                        val_pred = model(
                            val_seq,
                            edge_index,
                            batch_size=len(val_seq),
                            num_nodes=val_seq.shape[2],
                        )
                        val_loss = criterion(val_pred, val_tgt).item()
                else:
                    val_pred = model(
                        val_seq,
                        edge_index,
                        batch_size=len(val_seq),
                        num_nodes=val_seq.shape[2],
                    )
                    val_loss = criterion(val_pred, val_tgt).item()
            val_losses.append(val_loss)

            # Track best model for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[STATUS] Early stopping triggered at epoch {epoch+1}")
                    break

        # Progress logging every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            status = f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f}"
            if val_sequences is not None:
                status += f" | Val Loss: {val_loss:.6f}"
            print(status)

    # Restore best weights if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[INFO] Restored model weights from epoch with Val Loss: {best_val_loss:.6f}")

    print("--- Training Complete ---\n")
    return train_losses, val_losses