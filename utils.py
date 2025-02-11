import random
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import os
from PIL import Image
import pandas as pd

def seed_everything(seed=379647):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def metrics(y_true, y_prob):
    if type(y_prob) == torch.Tensor:
        pred = (y_prob>0.5).long()
    elif type(y_prob) == pd.Series:
        pred = (y_prob.to_numpy()>0.5).astype(int)
    else:
        raise ValueError
    roc_auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true,y_prob)
    balanced_accuracy = balanced_accuracy_score(y_true, pred)
    return roc_auc, auprc, balanced_accuracy

def plot_roc_auc(*results):
    fig, ax = plt.subplots()
    for (label, y_true, y_prob) in results:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # ax.set_title(f'ROC Curve - Epoch {epoch -1}')
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def highlight_words(tokens, word_importance, color="255, 255, 0"):
    highlighted_text = ""
    for word, importance in zip(tokens, word_importance):
        alpha = min(max(importance, 0), 1)
        highlighted_text += f'<span style="background-color: rgba({color}, {alpha:.2f});">{word}</span>'
    return highlighted_text.strip()

def html_to_image(html_str, output_file='temp.png'):
    from html2image import Html2Image
    hti = Html2Image(output_path='.')
    html_string = f"""
    <p style="font-family:Arial; font-size:16px; line-height:1.5; margin:0;">
        {html_str}
    </p>
    """
    hti.screenshot(html_str=html_string, size=(800, 200), save_as=output_file)
    return output_file

@torch.no_grad()
def plot_attention(model, sample, img_modalities, with_text, text, layer=4, device='cuda'):
    sample = {k: torch.from_numpy(v[None]) if type(v) == np.ndarray else v[None] for k,v in sample.items()}
    sample = {k: v.to(device) for k,v in sample.items()}
    out, attentions = model(sample, return_attention=True)
    image_size = sample[img_modalities[0]].shape[-1]
    # layer = -4
    # n_cols = len(img_modalities) if not with_text else len(img_modalities)+1
    # fig, axs = plt.subplots(1,n_cols,figsize=(n_cols*7,10))
    # if n_cols == 1:
    #     axs=[axs]

    ####################
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(4*7,15))
    gs = GridSpec(2, 4, height_ratios=[2, 1])  # Two rows: First is taller (2x), second is shorter (1x)

    # First row: 4 subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    # Second row: 1 subplot spanning all 4 columns
    ax5 = fig.add_subplot(gs[1, :])
    axs = [ax1,ax2,ax3,ax4,ax5]
    ####################
    
    mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
    std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
    for mod,ax in zip(img_modalities,axs): #,imgs):
        if mod == 'cxr':
            attn = attentions[mod][-1][0].cpu()
        else:
            attn = torch.cat([attentions[mod][i] for i in [0,1,2,3,4,11]], dim=0).sum(0).cpu()
        attn = attn.mean(0)
        attn = attn[0,1:]
        attn = attn.reshape(14,14)
        attn = F.interpolate(attn[None,None], size=(image_size,image_size), mode='bilinear', align_corners=False)[0,0]
        img = sample[mod][0].cpu()
        img = img * std + mean
        ax.imshow(img.permute(1,2,0))
        if mod == 'cxr':
            ax.imshow(attn, alpha=0.4)
        else:
            ax.imshow(attn, alpha=0.4, cmap='gray')
        ax.set_axis_off()
        ax.set_aspect(1.)
    if with_text:
        ax = axs[-1]
        attn = attentions['text'][-1][0].cpu()
        attn = attn.mean(0)
        attn = attn[0][1:]
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"][0])[1:]
        tokens = [t.replace('Ġ',' ').replace('Ċ','') for t in tokens]
        tokens = np.array(tokens)
        unwanted_tokens = ['<pad>', '<s>', '</s>'] # ',', '.', 
        mask = ~np.isin(tokens, unwanted_tokens)
        attn = attn[mask]
        tokens = tokens[mask]
        unwanted_attn_tokens = [',', '.']
        mask = np.isin(tokens, unwanted_attn_tokens)
        attn[mask] = 0
        attn = np.clip(attn, 0, np.percentile(attn, 95))
        attn = (attn - attn.min()) / (attn.max() - attn.min())
        html_string = highlight_words(tokens, attn)
        image_file = html_to_image(html_string)
        ax.imshow(Image.open(image_file))
        os.remove(image_file)
        ax.set_aspect(1.)
        ax.set_axis_off()

    fig.tight_layout()
    return fig, axs

def deploy_as_slurm_job(args, data_dir='/gpfs/bwfor/work/ws/hd_cn265-cfm'):
    import os
    import shutil
    from datetime import datetime
    ts = str(datetime.now().timestamp())
    code_dir = os.getcwd()
    if data_dir is None:
        data_dir = args.data_dir
    job_script = f'''
#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=46gb
module load devel/cuda

mkdir -p logs

conda init bash
source ~/.bash_profile
conda activate cfm

python main.py --model {args.model} --modalities {args.modalities} --with_text {args.with_text} --root {data_dir} --seed {args.seed} >> {code_dir}/logs/{ts}.log
'''
    fname = f'jobs/{ts}.slurm'
    with open(fname, 'w') as f:
        f.write(job_script)
    # os.system(fname)
    # shutil.remove(fname)