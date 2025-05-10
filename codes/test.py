import torch
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
#zheliceshi


def compute_metrics(preds, labels):
    """
    Compute accuracy and classification report.
    """
    accuracy = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds, output_dict=True, zero_division=0,digits=4)
    # Print reports with pandas, retaining 4 decimal places
    report_df = pd.DataFrame(report).transpose().round(4)
    print("\nClassification Report:\n", report_df)
    return accuracy, report

def test_model(model, test_loader, device, model_path="best_model.pt"):
    """
    Load the best model and evaluate it on the test set.
    """
    # Load the best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_preds, total_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            # Extract inputs
            comment_ids = batch['comment_ids'].to(device)
            comment_mask = batch['comment_mask'].to(device)
            explain_ids = batch.get('explain_ids', None)
            explain_mask = batch.get('explain_mask', None)
            if explain_ids is not None:
                explain_ids = explain_ids.to(device)
                explain_mask = explain_mask.to(device)
            dict_features = batch['dict_features'].to(device)
            dict_embed_vector = batch['dict_embed_vector'].to(device)
            targets = batch['targets'].to(device)

            # 正确解包模型输出
            logits, _ = model(
                comment_ids=comment_ids,
                comment_mask=comment_mask,
                explain_ids=explain_ids,
                explain_mask=explain_mask,
                dict_features=dict_features,
                dict_embed_vector=dict_embed_vector
            )

            total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            total_targets.extend(targets.cpu().numpy())

    # Compute metrics
    test_accuracy, test_report = compute_metrics(total_preds, total_targets)
    test_f1 = test_report['weighted avg']['f1-score']

    # Log metrics
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Classification Report:")
    print(classification_report(total_targets, total_preds, zero_division=0, digits=4))

    return test_accuracy, test_f1, test_report



# def test_model(model, test_loader, device, model_path="best_model.pt"):
#     """
#     Load the best model and evaluate it on the test set.
#     """
#     # Load the best model
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     total_preds, total_targets = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             # Extract inputs
#             comment_ids = batch['comment_ids'].to(device)
#             comment_mask = batch['comment_mask'].to(device)
#             explain_ids = batch.get('explain_ids', None)
#             explain_mask = batch.get('explain_mask', None)
#             if explain_ids is not None:
#                 explain_ids = explain_ids.to(device)
#                 explain_mask = explain_mask.to(device)
#             dict_features = batch['dict_features'].to(device)
#             targets = batch['targets'].to(device)
#
#             # Forward pass
#             logits = model(
#                 comment_ids=comment_ids,
#                 comment_mask=comment_mask,
#                 explain_ids=explain_ids,
#                 explain_mask=explain_mask,
#                 dict_features=dict_features,
#             )
#             total_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
#             total_targets.extend(targets.cpu().numpy())
#
#     # Compute metrics
#     test_accuracy, test_report = compute_metrics(total_preds, total_targets)
#     test_f1 = test_report['weighted avg']['f1-score']
#
#     # Log metrics
#     print(f"Test Accuracy: {test_accuracy:.4f}")
#     print(f"Test F1 Score: {test_f1:.4f}")
#     print("Classification Report:")
#     print(classification_report(total_targets, total_preds, zero_division=0, digits=4))
#
#     return test_accuracy, test_f1, test_report
