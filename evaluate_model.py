from screen_hash_model import ScreenHashModel
from evaluate import evaluate_model
import torch

def evaluate_saved_model(model_path, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScreenHashModel().to(device)
    model.load_state_dict(torch.load(model_path))
    
    mAP, μAP = evaluate_model(model, data_loader, device)
    print(f"模型评估结果 - mAP: {mAP:.4f}, μAP: {μAP:.4f}")
    return mAP, μAP