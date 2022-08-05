import os
import torch
from net_android import NET

cpt_dir = "/home/disk1/zhzhang/HolidayWork/exp/0"
cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
cpt = torch.load(cpt_fname, map_location='cpu')
model = NET()
model.load_state_dict(cpt["model_state_dict"])
model.eval()
example = torch.rand(16000)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./model.pt")