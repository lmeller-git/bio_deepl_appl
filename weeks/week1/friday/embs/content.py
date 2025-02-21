# Uncomment & execute once to download data

import itertools, numpy as np, pandas as pd, sklearn as sk, sklearn.preprocessing, sklearn.metrics, sklearn.naive_bayes

import matplotlib, matplotlib.colors, matplotlib.pyplot as plt, seaborn as sns

import torch, datasets, evaluate, transformers # Hugging Face libraries https://doi.org/10.18653/v1/2020.emnlp-demos.6

import py3Dmol # Visualising 3D structures; install with pip install py3Dmol
# Load the smallest version of the ESM-2 language model for masked token prediction

model_checkpoint = 'facebook/esm2_t6_8M_UR50D'

unmasker = transformers.pipeline(model=model_checkpoint)

esm_model = unmasker.model.esm

print("esm_model = ", esm_model)
n_esm_parameters = 0

for layer in esm_model.encoder.layer:

    n_esm_parameters += layer.intermediate.dense.weight.shape.numel() + layer.intermediate.dense.bias.shape.numel()

print(esm_model.num_parameters())

print(n_esm_parameters)
# Sequence for GCK glucokinase (P35557), see Figure 3F in https://www.science.org/doi/10.1126/science.adg7492#sec-5

seq = 'MLDDRARMEAAKKEKVEQILAEFQLQEEDLKKVMRRMQKEMDRGLRLETHEEASVKMLPTYVRSTPEGSEVGDFLSLDLGGTNFRVMLVKVGEGEEGQWSVKTKHQMYSIPEDAMTGTAEMLFDYISECISDFLDKHQMKHKKLPLGFTFSFPVRHEDIDKGILLNWTKGFKASGAEGNNVVGLLRDAIKRRGDFEMDVVAMVNDTVATMISCYYEDHQCEVGMIVGTGCNACYMEEMQNVELVEGDEGRMCVNTEWGAFGDSGELDEFLLEYDRLVDESSANPGQQLYEKLIGGKYMGELVRLVLLRLVDENLLFHGEASEQLRTRGAFETRFVSQVESDTGDRKQIYNILSTLGLRPSTTDCDIVRRACESVSTRAAHMCSAGLAGVINRMRESRSEDVMRITVGVDGSVYKLHPSFKERFHASVRRLTPSCEITFIESEEGSGRGAALVSAVACKKACMLGQ'

print(seq)



# Mask residue at pos

pos = 205

seq_masked = seq[:pos] + '<mask>' + seq[pos + 1:]

print("\n", seq_masked)



# Apply the ESM-2 model to the masked sequence

unmasker(seq_masked)
# Define residues with Alpha helix / Beta sheet / within a small-molecule binding site (inferred from structure using pymol)

resid_alpha_helix = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 112, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 238, 239, 240, 257, 258, 259, 267, 268, 269, 272, 273, 274, 275, 276, 277, 278, 279, 280, 288, 289, 290, 291, 292, 293, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 322, 323, 324, 325, 332, 333, 334, 335, 336, 337, 338, 339, 340, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 411, 412, 413, 414, 415, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464]

resid_beta_sheet = [72, 73, 74, 75, 76, 77, 78, 79, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 145, 146, 147, 148, 149, 150, 154, 155, 156, 157, 158, 161, 162, 163, 164, 201, 202, 203, 222, 223, 224, 225, 226, 230, 231, 232, 233, 234, 235, 236, 237, 250, 251, 252, 253, 254, 402, 403, 404, 405, 406, 407, 408, 409, 434, 435, 436, 437, 438, 439, 440]

resid_binding_sites = [61, 62, 63, 64, 65, 66, 151, 152, 153, 159, 168, 169, 204, 205, 206, 210, 211, 214, 218, 220, 221, 225, 229, 230, 231, 235, 250, 254, 256, 258, 287, 290, 451, 452, 455, 456, 459]

print(len(resid_alpha_helix), 'residues in alpha helix')

print(len(resid_beta_sheet), 'residues in as beta sheets')

print(len(resid_binding_sites), 'residues in binding sites')



cmap = matplotlib.colormaps['cividis']

color_alpha = matplotlib.colors.to_hex(cmap(255))

color_beta = matplotlib.colors.to_hex(cmap(175))

color_binding = matplotlib.colors.to_hex(cmap(75))

color_null = matplotlib.colors.to_hex(cmap(0))

view = py3Dmol.view()

view.addModel(open('data/AF-P35557-F1-model_v4.pdb', 'r').read(), 'pdb')

view.setStyle({

    'cartoon': {

        'colorscheme': {

            'prop': 'resi',

            'map': { resi + 1: color_alpha if resi + 1 in resid_alpha_helix else color_beta if resi + 1 in resid_beta_sheet else color_binding if resi + 1 in resid_binding_sites else color_null for resi in range(len(seq)) },

        },

        'arrows': True,

    }

})

view.zoomTo()

view.show()
# Run the raw ESM-2 model on the raw unmasked sequence, capture full output and internal state

tokens = unmasker.tokenizer(seq, return_tensors='pt').to('cuda')

outputs = unmasker.model(tokens['input_ids'], attention_mask=tokens['attention_mask'], output_hidden_states=True,)
dir(outputs)
def get_layer_from_output(layer_id):

    return outputs.hidden_states[layer_id]



def get_embedding_from_output(layer_id, embedding_id):

    layer = get_layer_from_output(layer_id)

    #print(dir(layer))

    print(layer.shape)

    return layer[-1, :, embedding_id].cpu().detach().numpy()#[-1, 1] * int(len(seq) / 2)



get_embedding_from_output(2, 3)
def get_colors(values):

    rescaled = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))(values)

    pos_col = {i + 1: matplotlib.colors.to_hex(cmap(val)) for i, val in enumerate(rescaled) }

    return pos_col



view = py3Dmol.view()

view.addModel(open('data/AF-P35557-F1-model_v4.pdb', 'r').read(), 'pdb')

view.setStyle({

    'cartoon': {

        'colorscheme': {

            'prop': 'resi',

            'map': get_colors(get_embedding_from_output(5, 22)),

        },

        'arrows': True,

    }

})

view.zoomTo()

view.show()
y_true_alpha_helix = [ resi + 1 in resid_alpha_helix for resi in range(len(seq)) ]

y_true_beta_sheet = [ resi + 1 in resid_beta_sheet for resi in range(len(seq)) ]

y_true_binding_sites = [ resi + 1 in resid_binding_sites for resi in range(len(seq)) ]



scores_alpha_helix = []

scores_beta_sheet = []

scores_binding_sites = []



auc_threshold = .7

for layer_id in range(7):

    print('layer', layer_id)

    X = get_layer_from_output(layer_id)

    #print(len(y_true_alpha_helix), X.shape)

    #print(X[:, 1:-1, 0], len(y_true_alpha_helix))

    auc_alpha_helix = np.array([sk.metrics.roc_auc_score(y_true=y_true_alpha_helix, y_score=X[:, 1:-1,i].cpu().detach().reshape(-1, 1).numpy(), multi_class = "raise") for i in range(X.shape[2]) ])

    score_alpha_helix = sum(auc_alpha_helix > auc_threshold)

    print('Best auc_alpha_helix:', max(auc_alpha_helix), np.argmax(auc_alpha_helix))



    # ...

    auc_beta_sheet = np.array([sk.metrics.roc_auc_score(y_true=y_true_beta_sheet, y_score=X[:, 1:-1,i].cpu().detach().reshape(-1, 1).numpy(), multi_class = "raise") for i in range(X.shape[2]) ])

    score_beta_sheet = sum(auc_beta_sheet > auc_threshold)

    print('Best auc_beta_sheet:', max(auc_beta_sheet), np.argmax(auc_beta_sheet))



    auc_binding = np.array([sk.metrics.roc_auc_score(y_true=y_true_binding_sites, y_score=X[:, 1:-1,i].cpu().detach().reshape(-1, 1).numpy(), multi_class = "raise") for i in range(X.shape[2]) ])

    score_binding = sum(auc_binding > auc_threshold)

    print('Best auc_binding_site:', max(auc_binding), np.argmax(auc_binding))



    scores_alpha_helix.append(score_alpha_helix)

    scores_beta_sheet.append(score_beta_sheet)

    scores_binding_sites.append(score_binding)

    # ...



layers = [1, 2, 3, 4, 5, 6, 7]

centre_alpha_helix = np.average(layers, weights=scores_alpha_helix)

centre_beta_sheet = np.average(layers, weights=scores_beta_sheet)

centre_binding_sites = np.average(layers, weights=scores_binding_sites)





print(scores_alpha_helix, centre_alpha_helix)
plt.figure(figsize=(4, 4))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

plt.subplot(3, 1, 1)

plt.bar(layers, scores_alpha_helix)

plt.axvline(centre_alpha_helix, color='tab:red', linewidth=3)

plt.gca().set_ylabel('Alpha helix')



plt.subplot(3, 1, 2)

# ...

plt.bar(layers, scores_beta_sheet)

plt.axvline(centre_beta_sheet, color='tab:red', linewidth=3)

plt.gca().set_ylabel('Beta sheet')



plt.subplot(3, 1, 3)

# ...

plt.bar(layers, scores_binding_sites)

plt.axvline(centre_binding_sites, color='tab:red', linewidth=3)

plt.gca().set_ylabel('Binding site')

plt.gca().set_xlabel('Layer')
