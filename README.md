# DynamicBind
Source code for the *Nature Communications* paper [DynamicBind: predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model](https://www.nature.com/articles/s41467-024-45461-2).

DynamicBind recovers ligand-specific conformations from unbound protein structures (e.g. AF2-predicted structures), promoting efficient transitions between different equilibrium states.

![](dynbind.gif)

# Setup Environment

Create environment dynamicbind:
```commandline
conda create -n dynamicbind python=3.10
conda activate dynamicbind
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install rdkit
pip install pyyaml biopython
pip install e3nn fair-esm spyrmsd
pip install pandas
pip install tqdm
 ```

Create another environment for structural Relaxation.
```commandline
conda create --name relax python=3.8
conda activate relax
pip install openmm
conda install -c conda-forge pdbfixer
pip install compilers biopython
pip install tqdm
pip install scipy
pip install pandas
pip install rdkit
pip install networkx
```

# Download pretrained ESM models
```commandline
mkdir esm_models
cd esm_models/
mkdir checkpoints
cd checkpoints/
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

# Download checkpoints
Download and unzip the workdir.zip containing the model checkpoint form https://zenodo.org/records/10137507, 
v2 is contained here https://zenodo.org/records/10183369.

# Inference

## Dynamic Docking
By default: 40 poses will be predicted, poses will be ranked (rank1 is the best-scoring pose, rank40 the lowest), 
relax processes are included.

#### Inputs:
1. **Protein (PDB File):** `protein.pdb` 
   - Automatically cleaned to remove non-standard amino acids, water molecules, or small molecules.
2. **Ligand (CSV File):** `ligand.csv` 
   - Must contain a column named 'ligand' listing smiles.
3. **Number of Animations:** 
   - outputs intermediate pkl data, not the final animation PDB. (After `--savings_per_complex`, default is 40)
4. **Frames in Animation/inference_steps:** 
   - default is 20.

#### Additional Options:
- `--header`: Name of the result folder.
- `--device`: GPU device ID.
- `--python`: Python environment for inference.
- `--relax_python`: Python environment for relaxation.
- `--num_workers`: Number of processes for final output relaxation.

#### Example Command:
```bash
python run_single_protein_inference.py data/origin-1qg8.pdb data/1qg8_input.csv \
--savings_per_complex 40 \
--inference_steps 20 \
--header test --device 0 \
--python /home/ruofan/anaconda3/envs/dynamicbind/bin/python \
--relax_python /home/ruofan/anaconda3/envs/relax/bin/python \
--movie
```


[//]: # (### Docking Outputs)

[//]: # ()
[//]: # (The results of the docking step, typically found in the `results/test` folder, include:)

[//]: # ()
[//]: # ()
[//]: # (1. **Affinity Score for Each Complex**: `affinity_prediction.csv`)

[//]: # ()
[//]: # (2. **Pose Score and Conformation of Each Animation**: Example files like `rank1_ligand_lddt0.63_affinity5.67_relaxed.sdf` &#40;where 0.63 is the pose score&#41; and corresponding protein `.pdb` files.)

[//]: # ()
[//]: # (3. **Data for Animation Generation**: Such as `rank1_reverseprocess_data_list.pkl` and `rank2_reverseprocess_data_list.pkl`.)

[//]: # ()
[//]: # ()
[//]: # (## Movie Generation)

[//]: # ()
[//]: # (Inputs:)

[//]: # ()
[//]: # (1. **Data from Docking Output**: Indicated by paths like `results/test/index0_idx_0/`. )

[//]: # ()
[//]: # (The notation "1+2" implies that movies for rank1 and rank2 poses are needed.)

[//]: # ()
[//]: # (2. **Number of Animations**: Specified by the user &#40;default is "1"&#41;.)

[//]: # ()
[//]: # ()
[//]: # (#### Example command for generating movies:)

[//]: # ()
[//]: # (```bash)

[//]: # ()
[//]: # (python movie_generation.py results/test/index0_idx_0/ 1+2 --python /home/ruofan/anaconda3/envs/dynamicbind/bin/python --relax_python /home/ruofan/anaconda3/envs/relax/bin/python)

[//]: # ()
[//]: # (```)

[//]: # (Outputs:)
[//]: # (- **Final Animation PDB Files**: Located in `results/test_1qg8/index0_idx_0/`, )
[//]: # (with files like `rank1_receptor_reverseprocess_relaxed.pdb` and `rank1_ligand_reverseprocess_relaxed.pdb`.)

[//]: # (## High-Throughput Screening &#40;HTS&#41;)

[//]: # (Example command for HTS:)

[//]: # (```bash)

[//]: # (python run_single_protein_inference.py protein.pdb ligand.csv )

[//]: # (--hts )

[//]: # (--savings_per_complex 3 )

[//]: # (--inference_steps 20 )

[//]: # (--header test )

[//]: # (--device $1 )

[//]: # (--python /path/to/dynamicbind/python )

[//]: # (--relax_python /path/to/relax/python)

[//]: # (```)

[//]: # ()
[//]: # (HTS Output files:)

[//]: # (- `complete_affinity_prediction.csv`)

[//]: # (- `affinity_prediction.csv`)

[//]: # ()
[//]: # (# Training and testing Dataset)

[//]: # ( https://zenodo.org/records/10429051)

[//]: # ()
[//]: # (# Reference)

[//]: # (```bibtex)

[//]: # (@article{lu2024dynamicbind,)

[//]: # (  title={DynamicBind: predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model},)

[//]: # (  author={Lu, Wei and Zhang, Jixian and Huang, Weifeng and Zhang, Ziqiao and Jia, Xiangyu and Wang, Zhenyu and Shi, Leilei and Li, Chengtao and Wolynes, Peter G and Zheng, Shuangjia},)

[//]: # (  journal={Nature Communications},)

[//]: # (  volume={15},)

[//]: # (  number={1},)

[//]: # (  pages={1071},)

[//]: # (  year={2024},)

[//]: # (  publisher={Nature Publishing Group UK London})

[//]: # (})

[//]: # (```)


