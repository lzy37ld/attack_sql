# Assuming two processes, with a batch size of 5 on a dataset with 9 samples
import torch
from accelerate import Accelerator

accelerator = Accelerator()
dataloader = torch.utils.data.DataLoader(range(8), batch_size=2)
dataloader = accelerator.prepare(dataloader)
print(len(dataloader))
# batch = next(iter(dataloader))
# gathered_items = accelerator.gather_for_metrics(batch)
# len(gathered_items)