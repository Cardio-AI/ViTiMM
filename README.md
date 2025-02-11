# Visual Prompt Engineering for Multimodal and Irregularly Sampled Medical Data

Malte Tölle, Mohamad Scharaf, Samantha Fischer, Christoph Reich, Silav Zeid, Christoph Dieterich, Benjamin Meder, Norbert Frey, Philipp Wild, Sandy Engelhardt

Paper link: 

## Abstract

A multitude of examinations are conducted to assess a patient's health, with each modality contributing unique information that collectively creates comprehensive understanding.
These assessments include temporal data with varying sampling rates as well as single value measurements, interventions like medications, or imaging modalities.
While physicians are able to process different information easily, neural networks need specific modeling for each modality complicating the training procedure.
We demonstrate that this complexity can be significantly reduced by visualizing all information as images along with unstructured text and subsequently training a conventional vision-text transformer.
Our approach, Vision Transformer for irregular sampled Multi-modal Measurements (ViTiMM), not only simplifies data preprocessing and modeling but also outperforms current state-of-the-art methods in predicting in-hospital mortality and phenotyping, as evaluated on 6,175 patients from the MIMIC-IV dataset.
The modalities include patient's clinical measurements, medications, X-ray images, and electrocardiography scans. % characteristics, conditions, and 
We hope our work inspires advancements in multi-modal medical AI by reducing the training complexity to (visual) prompt engineering, thus lowering entry barriers and enabling no-code solutions for training.

## Usage

After downloading the MIMIC datasets all plots can be created with the plot_[labs,ecgs,meds].ipynb files.

MIMIC-IV: https://physionet.org/content/mimiciv/3.1/

MIMIC-CXR: https://physionet.org/content/mimic-cxr-jpg/2.1.0/

MIMIC-IC-ECG: https://physionet.org/content/mimic-iv-ecg/1.0/


Place the `runs` folder in this directory, the `data` directory can have an arbitrary location.

Training can be performed with:
```
python main.py \
    --task [inhospital_mortality,phenotyping] \
    --model [swin,vit] \
    --modalities lab med cxr ecg \
    [--with_text] \
    --root PATH_TO_DATA \
    --n_epochs 3 \
    --weight_decay 3e-8 \
    --lrs 1e-5 5e-6 1e-6 \
    --batch_size 4 \
    [--ckpt PATH_TO_CKPT] \
    --seed 0
```

## BibTeX

```
@misc{toelle2025vitimm,
    title={Arbitrary Data as Images: Fusion of Patient Data Across Modalities and Irregular Intervals with Vision Transformers},
    author={T{\"o}lle, Malte and Scharaf, Mohamad and Fischer, Samantha and Reich, Christoph and Zeid, Silav and Dieterich, Christoph and Meder, Benjamin and Frey, Norbert and Wild, Philipp and Engelhardt, Sandy},
    year={2025},
    doi={10.48550/arXiv.2501.18237}
}
```

## Contact

Malte Tölle<br>
[malte.toelle@med.uni-heidelberg.de](mailto:malte.toelle@med.uni-heidelberg.de)<br>
[@maltetoelle](https://x.com/maltetoelle)<br>

Group Artificial Intelligence in Cardiovascular Medicine (AICM)<br>
Heidelberg University Hospital<br>
Im Neuenheimer Feld 410, 69120 Heidelberg, Germany