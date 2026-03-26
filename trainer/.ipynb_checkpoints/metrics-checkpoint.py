import torch
from torchmetrics.classification import BinaryMatthewsCorrCoef, BinaryAUROC, BinaryAveragePrecision


def setup_metrics(args):
    """Setup metrics based on problem type and specified metrics list."""
    metrics_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Handle case where metrics is None
    if args.metrics is None:
        args.metrics = ['loss']
    
    for metric_name in args.metrics:
        if args.problem_type == 'single_label_classification' and (args.num_labels == 1 or args.num_labels == 2):
            metric_config = _setup_binary_metrics(metric_name, device)
            if metric_config:
                metrics_dict[metric_name] = metric_config['metric']

    # Add loss to metrics if it's the monitor metric
    if args.monitor == 'loss':
        metrics_dict['loss'] = 'loss'

    return metrics_dict


def _setup_binary_metrics(metric_name, device):
    metrics_config = {
        'mcc': {
            'metric': BinaryMatthewsCorrCoef().to(device),
        },
        'auroc': {
            'metric': BinaryAUROC().to(device),
        },
        'aupr': {
            'metric': BinaryAveragePrecision().to(device),
        }
    }
    return metrics_config.get(metric_name)
