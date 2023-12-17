import torch
import os
import numpy as np
from dataset import PISToN_dataset
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm


import sys
# sys.path.append("..")
import pickle
import torch
import numpy as np
#from networks.PIsToN_multiAttn import PIsToN_multiAttn
#from networks.ViT_pytorch import get_ml_config

#MODEL_DIR='./saved_models/'
#MODEL_NAME='PIsToN_multiAttn_contrast'

## Paremeters of the original training:
#params = {'dim_head': 16,
#          'hidden_size': 16,
#          'dropout': 0,
#          'attn_dropout': 0,
#          'n_heads': 8,
#          'patch_size': 4,
#          'transformer_depth': 8}

#device=torch.device("cuda")
#device=torch.device("cpu")

#model_config=get_ml_config(params)
#model = PIsToN_multiAttn(model_config, img_size=32,
#                        num_classes=2).float().to(device)

#model.load_state_dict(torch.load(MODEL_DIR + '/{}.pth'.format(MODEL_NAME), map_location=device))
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#n_params = sum([np.prod(p.size()) for p in model_parameters])

#print(f"Loaded PiSToN model with {n_params} trainable parameters.")

import PyIO
import PyPluMA
import pickle

class AttentionMapPlugin:
    def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
    def run(self):
        pass
    def output(self, outputfile):
      inputfile = open(PyPluMA.prefix()+"/"+self.parameters["model"], 'rb')
      model = pickle.load(inputfile)

      start = time()
      torch.set_num_threads(1)
      device=torch.device("cpu")

      GRID_DIR = PyPluMA.prefix()+"/"+self.parameters["grid"]#'data/masif_test/prepare_energies_16R/07-grid/'
      if ('ppi' not in self.parameters):
        ppi_list = os.listdir(GRID_DIR)
        ppi_list = [x.split('.npy')[0] for x in ppi_list if 'resnames' not in x and '.ref' not in x]


      else:
        ppi_list = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["ppi"])

      labels = [0 if 'neg' in x else 1 for x in ppi_list]
      print(f"Extracted {len(ppi_list)} complexes.")
      print(f"{np.sum(labels)} acceptable and {len(labels) -np.sum(labels) } incorrect.")
      masif_test_dataset = PISToN_dataset(GRID_DIR, ppi_list)



      masif_test_loader = DataLoader(masif_test_dataset, batch_size=1, shuffle=False, pin_memory=False)

      all_outputs = [] # output score
      all_attn = [] # output attention map
      predicted_labels = [] #predicted label (0 for negative and 1 for positive)

      with torch.no_grad():
       for instance in tqdm(masif_test_loader):
        grid, all_energies = instance
        grid = grid.to(device)
        all_energies = all_energies.float().to(device)
        model = model.to(device)
        output2, attn = model(grid, all_energies)
        all_outputs.append(output2)
        all_attn.append(attn)
        if float(output2)<0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

      output2 = torch.cat(all_outputs, axis=0)
      print(len(output2))
      print(f"Total inference time: {time() - start} sec")
      picklejar = PyPluMA.prefix()+"/"+self.parameters["picklejar"]
      outfile = open(picklejar+"/"+outputfile[outputfile.rfind('/')+1:]+".torches.pkl", "wb")
      attnfile = open(picklejar+"/"+outputfile[outputfile.rfind('/')+1:]+".attn.pkl", "wb")
      pickle.dump(output2, outfile)
      pickle.dump(all_attn, attnfile)

      predfile = open(outputfile+".predicted.txt", "w")
      for i in range(len(predicted_labels)):
          predfile.write(str(predicted_labels[i])+"\n")

      ppifile = open(outputfile+".ppi.txt", "w")
      for i in range(len(masif_test_dataset.ppi_list)):
          ppifile.write(str(masif_test_dataset.ppi_list[i])+"\n")
