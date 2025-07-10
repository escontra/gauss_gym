
def print_stats_per_param_grup(model, optimizer):
  print("\nImage Encoder Parameter Gradients and Learning Rates:")
  for name, param in model.named_parameters():
    if param.grad is None:
      print(f"{name}: NO GRADIENTS")
    else:
      lr, param_group_name = None, None
      for param_group in optimizer.param_groups:
        for sub_param in param_group['params']:
          if sub_param is param:
            lr = param_group['lr']
            param_group_name = param_group['name']
      if lr is None:
        raise ValueError(f"{name}: NO PARAM GROUP FOUND")

      print(f"{param_group_name}/{name}:")
      print(f"  Learning Rate: {lr}")
      print(f"  Gradient Mean: {param.grad.mean().item():.6f}")
      print(f"  Gradient Std: {param.grad.std().item():.6f}")
      print(f"  Gradient Min: {param.grad.min().item():.6f}")
      print(f"  Gradient Max: {param.grad.max().item():.6f}")
