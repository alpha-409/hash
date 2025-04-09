# main.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import random
import argparse
import time

# --- Import custom modules ---
from utils import load_data # Assuming you saved it as utils.py or data_loader.py
from model import ScreenHashNet, HashingLoss
from dataset import ImageHashingDataset, collate_fn
from evaluation import calculate_metrics # Updated import

# --- Configuration ---
# (parse_args function remains the same)
def parse_args():
    parser = argparse.ArgumentParser(description='ScreenHashNet Training and Evaluation')
    parser.add_argument('--dataset', type=str, default='copydays', help='Dataset name (e.g., copydays, scid)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Root directory for datasets')
    parser.add_argument('--hash-bits', type=int, default=64, help='Number of hash bits (L)')
    parser.add_argument('--cp-rank', type=int, default=32, help='CP decomposition rank (R)')
    parser.add_argument('--backbone', type=str, default='resnet18', help='DFE backbone (e.g., resnet18, resnet34, resnet50)')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained DFE backbone')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', help='Do not use pretrained DFE backbone')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate') # Lower LR often needed for fine-tuning
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--margin', type=float, default=0.4, help='Margin for similarity loss') # Adjust this
    parser.add_argument('--lambda-sim', type=float, default=1.0, help='Weight for similarity loss')
    parser.add_argument('--lambda-quant', type=float, default=0.5, help='Weight for quantization loss')
    parser.add_argument('--lambda-balance', type=float, default=0.1, help='Weight for balance loss')
    parser.add_argument('--num-workers', type=int, default=None, help='DataLoader workers (default: auto)')
    parser.add_argument('--eval-epoch', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save checkpoints and results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--simulate-images', action='store_true', default=True, help='Generate dummy images if missing')
    parser.add_argument('--no-simulate-images', action='store_false', dest='simulate_images')

    return parser.parse_args()


# (set_seed function remains the same)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# (train_one_epoch function remains the same)
def train_one_epoch(model, loader, optimizer, criterion, device, positive_pairs_lookup):
    model.train()
    total_loss_accum = 0.0
    sim_loss_accum = 0.0
    quant_loss_accum = 0.0
    balance_loss_accum = 0.0
    num_batches = 0

    progress_bar = tqdm(loader, desc='Training', leave=False)
    for images, S_matrix, _ in progress_bar: # Ignore original indices here
        images = images.to(device)
        S_matrix = S_matrix.to(device) # Move S to device

        optimizer.zero_grad()
        H, tilde_H = model(images)
        total_loss, loss_dict = criterion(H, tilde_H, S_matrix)

        if torch.isnan(total_loss):
            print("Warning: NaN loss detected. Skipping batch.")
            continue # Skip backpropagation if loss is NaN

        total_loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_accum += loss_dict['total_loss'].item()
        sim_loss_accum += loss_dict['loss_sim'].item()
        quant_loss_accum += loss_dict['loss_quant'].item()
        balance_loss_accum += loss_dict['loss_balance'].item()
        num_batches += 1

        progress_bar.set_postfix({
            'Loss': f"{loss_dict['total_loss'].item():.4f}",
            'Sim': f"{loss_dict['loss_sim'].item():.4f}",
            'Quant': f"{loss_dict['loss_quant'].item():.4f}",
            'Bal': f"{loss_dict['loss_balance'].item():.4f}"
        })

    avg_total_loss = total_loss_accum / num_batches if num_batches > 0 else 0
    avg_sim_loss = sim_loss_accum / num_batches if num_batches > 0 else 0
    avg_quant_loss = quant_loss_accum / num_batches if num_batches > 0 else 0
    avg_balance_loss = balance_loss_accum / num_batches if num_batches > 0 else 0

    return avg_total_loss, avg_sim_loss, avg_quant_loss, avg_balance_loss


# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model, data_dict, device, hash_bits, batch_size=128): # Added hash_bits arg
    model.eval()

    query_images = data_dict['query_images']
    db_images = data_dict['db_images']
    gnd = data_dict['gnd']
    qimlist = data_dict['qimlist']
    imlist = data_dict['imlist']

    num_query = len(query_images) if isinstance(query_images, (list, torch.Tensor)) else 0
    num_db = len(db_images) if isinstance(db_images, (list, torch.Tensor)) else 0

    # Handle empty datasets gracefully
    if num_query == 0 or num_db == 0:
        print("Warning: Query or Database image set is empty. Skipping evaluation.")
        return {'mAP': 0.0, 'muAP': 0.0}

    query_hashes = torch.zeros(num_query, hash_bits).to(device) # Use hash_bits argument
    db_hashes = torch.zeros(num_db, hash_bits).to(device) # Use hash_bits argument

    print("Generating Query Hashes...")
    for i in tqdm(range(0, num_query, batch_size)):
        end = min(i + batch_size, num_query)
        # Ensure batch is a tensor before sending to device
        batch_data = query_images[i:end]
        if isinstance(batch_data, list):
             batch_data = torch.stack(batch_data) # Stack if it's still a list of tensors
        batch = batch_data.to(device)
        H, _ = model(batch)
        query_hashes[i:end] = H

    print("Generating Database Hashes...")
    for i in tqdm(range(0, num_db, batch_size)):
        end = min(i + batch_size, num_db)
        # Ensure batch is a tensor
        batch_data = db_images[i:end]
        if isinstance(batch_data, list):
             batch_data = torch.stack(batch_data)
        batch = batch_data.to(device)
        H, _ = model(batch)
        db_hashes[i:end] = H

    # Prepare ground truth
    query_labels_map = list(range(num_query))
    db_labels_map = list(range(num_db))

    gnd_map = []
    for q_idx in range(num_query):
        relevant_indices = set()
        if q_idx < len(gnd):
            q_gnd_info = gnd[q_idx]
            # Define relevance based on these keys (adjust if necessary)
            relevant_keys = ['strong', 'crops', 'jpegqual']
            for key in relevant_keys:
                if key in q_gnd_info:
                    # Ensure indices are integers
                    try:
                        current_indices = set(map(int, q_gnd_info[key]))
                        relevant_indices.update(current_indices)
                    except (ValueError, TypeError) as e:
                         print(f"Warning: Could not process indices for query {q_idx}, key {key}. Error: {e}. Value: {q_gnd_info[key]}")

        else:
             print(f"Warning: Query index {q_idx} out of bounds for ground truth list (length {len(gnd)}).")
        gnd_map.append(relevant_indices)

    print("Calculating Metrics (mAP, μAP)...")
    # Ensure hashes are on CPU if calculate_metrics expects CPU tensors
    metrics = calculate_metrics(query_hashes.cpu(), db_hashes.cpu(),
                                query_labels=query_labels_map,
                                db_labels=db_labels_map,
                                gnd_truth=gnd_map,
                                top_k=None) # Calculate metrics over all results

    return metrics # Return the dictionary


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # --- Device Setup ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"log_{args.dataset}_{args.hash_bits}bit.txt")

    def log_message(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    log_message("--- Configuration ---")
    for arg, value in vars(args).items():
        log_message(f"{arg}: {value}")
    log_message("---------------------")


    # --- Load Data ---
    log_message(f"Loading dataset: {args.dataset}...")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    try:
        # Make sure load_data is imported correctly (e.g., from utils)
        data_dict = load_data(args.dataset, args.data_dir, data_transform,
                               simulate_images=args.simulate_images,
                               num_workers=args.num_workers)
    except FileNotFoundError as e:
        log_message(f"Error loading data: {e}")
        log_message("Please ensure the dataset is correctly placed in the specified data directory.")
        exit(1)
    except Exception as e:
        log_message(f"An unexpected error occurred during data loading: {e}")
        exit(1)

    log_message("Data loaded successfully.")

    # --- Create Dataset and DataLoader ---
    train_dataset = ImageHashingDataset(data_dict, dataset_type='db')
    positive_pairs_lookup = train_dataset.positive_pairs_db_indices
    collate_wrapper = lambda batch: collate_fn(batch, positive_pairs_lookup)

    # Determine optimal num_workers if not specified
    num_workers = args.num_workers if args.num_workers is not None else min(os.cpu_count() // 2, 8) # Heuristic

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_wrapper,
        pin_memory=use_cuda, # Use pin_memory if using CUDA
        persistent_workers=True if num_workers > 0 else False # Can speed up loading
    )
    log_message(f"Training DataLoader created with batch size {args.batch_size}, workers {num_workers}.")


    # --- Initialize Model, Criterion, Optimizer ---
    log_message("Initializing model...")
    model = ScreenHashNet(
        hash_bits=args.hash_bits,
        cp_rank=args.cp_rank,
        dfe_backbone=args.backbone,
        dfe_pretrained=args.pretrained,
        use_cuda=use_cuda
    ).to(device)
    log_message(f"Model: ScreenHashNet ({args.backbone} backbone, {args.hash_bits} bits)")
    log_message(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    criterion = HashingLoss(
        lambda_sim=args.lambda_sim,
        lambda_quant=args.lambda_quant,
        lambda_balance=args.lambda_balance,
        sim_margin=args.margin
    ).to(device)
    log_message(f"Loss: HashingLoss (margin={args.margin}, sim={args.lambda_sim}, quant={args.lambda_quant}, bal={args.lambda_balance})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    log_message(f"Optimizer: AdamW (lr={args.lr}, wd={args.wd})")

    # --- Training Loop ---
    log_message("\n--- Starting Training ---")
    best_map = 0.0
    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        log_message(f"\nEpoch {epoch}/{args.epochs}")

        avg_loss, avg_sim, avg_quant, avg_bal = train_one_epoch(
            model, train_loader, optimizer, criterion, device, positive_pairs_lookup
        )

        log_message(f"Epoch {epoch} Training Summary:")
        log_message(f"  Avg Total Loss: {avg_loss:.4f}")
        log_message(f"  Avg Sim Loss:   {avg_sim:.4f}")
        log_message(f"  Avg Quant Loss: {avg_quant:.4f}")
        log_message(f"  Avg Bal Loss:   {avg_bal:.4f}")
        log_message(f"  Epoch Time: {time.time() - epoch_start_time:.2f}s")

        # --- Evaluation ---
        if epoch % args.eval_epoch == 0 or epoch == args.epochs:
            log_message(f"\n--- Evaluating Epoch {epoch} ---")
            eval_start_time = time.time()
            # Pass hash_bits to evaluate function
            metrics = evaluate(model, data_dict, device, args.hash_bits, batch_size=args.batch_size * 2)
            current_map = metrics['mAP']
            current_muap = metrics['muAP'] # Get muAP from the returned dict
            log_message(f"Epoch {epoch} mAP: {current_map:.4f}, μAP: {current_muap:.4f}") # Log both
            log_message(f"Evaluation Time: {time.time() - eval_start_time:.2f}s")

            # Save best model (based on mAP)
            if current_map > best_map:
                best_map = current_map
                log_message(f"*** New Best mAP: {best_map:.4f} ***")
                checkpoint_path = os.path.join(args.output_dir, f"best_model_{args.dataset}_{args.hash_bits}bit.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'map': best_map, # Save best mAP
                    'muap': current_muap, # Also save corresponding muAP
                    'args': args
                }, checkpoint_path)
                log_message(f"Best model saved to {checkpoint_path}")

            # Save latest model
            latest_checkpoint_path = os.path.join(args.output_dir, f"latest_model_{args.dataset}_{args.hash_bits}bit.pth")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'map': current_map, # Save current mAP
                    'muap': current_muap, # Save current muAP
                    'args': args
                }, latest_checkpoint_path)

    total_training_time = time.time() - start_train_time
    log_message("\n--- Training Finished ---")
    log_message(f"Total Training Time: {total_training_time / 3600:.2f} hours")
    log_message(f"Best mAP achieved: {best_map:.4f}")
    log_message(f"Results and checkpoints saved in: {args.output_dir}")
    log_message(f"Log file saved to: {log_file}")