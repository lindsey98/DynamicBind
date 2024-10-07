import torch
import os
import subprocess
import argparse

def do(cmd, get=False, show=True):
    if get:
        out = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0].decode()
        if show:
            print(out, end="")
        return out
    else:
        return subprocess.Popen(cmd, shell=True).wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run virtual screening")

    parser.add_argument('--samples_per_complex', type=int, default=10, help='num of samples data generated.')
    parser.add_argument('--savings_per_complex', type=int, default=10, help='num of samples data saved for movie generation.')
    parser.add_argument('--inference_steps', type=int, default=20, help='num of coordinate updates. (movie frames)')
    parser.add_argument('--header', type=str, default='test', help='informative name used to name result folder')
    parser.add_argument('--results', type=str, default='results', help='result folder.')

    parser.add_argument('--seed', type=int, default=42, help='set seed number')

    parser.add_argument('--python', type=str,
                        default='/home/ruofan/anaconda3/envs/dynamicbind/bin/python',
                        help='point to the python in dynamicbind env.')
    parser.add_argument('--ligandCSVDirs', default='./datasets/ligand_csv/', help="where to save the preprocessed protein to ligand csv files")

    # the following are processed by tankbind
    parser.add_argument('--ligandDirs', default="/home/ruofan/git_space/TankBind/datasets/drugbank", help='Directory for ligand files')
    parser.add_argument('--ligandRdkitDirs', default="/home/ruofan/git_space/TankBind/datasets/drugbank_rdkit", help='Where to save the ligand RDKit files')
    parser.add_argument('--proteinDirsUni', default="/home/ruofan/git_space/TankBind/datasets/protein_315", help='Directory for protein files (unfied)')

    args = parser.parse_args()

    os.makedirs(args.ligandCSVDirs, exist_ok=True)

    header = args.header

    protein_dict = torch.load('/home/ruofan/git_space/TankBind/datasets/protein_315.pt') # this is processed by TankBind

    '''Preprocess to csv files, the csv files have two columns: ligand.sdf paths and protein.pdb paths'''
    # for proteinName in list(protein_dict.keys()):
    #     ligandFile_list = []
    #     for ligand in tqdm(os.listdir(args.ligandDirs)):
    #         ligandName = ligand.split('.sdf')[0]
    #         ligandFile = f"{args.ligandRdkitDirs}/{ligandName}_from_rdkit.sdf"
    #         ligandFile_list.append(ligandFile)
    #
    #     ligands = pd.DataFrame({"ligand":ligandFile_list,
    #                             "protein_path":[os.path.join(args.proteinDirsUni, proteinName+".pdb")]*len(ligandFile_list)})
    #     ligands.to_csv(os.path.join(args.ligandCSVDirs, f"{proteinName}.csv"), index=False)


    #
    # '''Convert proteins to fasta'''
    # for proteinName in list(protein_dict.keys()):
    #     if not os.path.exists(f"data/prepared_for_esm_{header}_{proteinName}.fasta"):
    #         ligandFile_with_protein_path = os.path.join(args.ligandCSVDirs, f"{proteinName}.csv")
    #         cmd = f"{args.python} ./datasets/esm_embedding_preparation.py " \
    #               f"--protein_ligand_csv {ligandFile_with_protein_path} " \
    #               f"--out_file data/prepared_for_esm_{header}_{proteinName}.fasta"
    #         do(cmd)
    #
    # '''Convert protein fasta to esm embeddings (this may take time)'''
    # for proteinName in list(protein_dict.keys()):
    #     if (not os.path.exists(f"data/esm2_output_{proteinName}")) or len(os.listdir(f"data/esm2_output_{proteinName}")) == 0:
    #         cmd = f"{args.python} ./esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_{header}_{proteinName}.fasta data/esm2_output_{proteinName} " \
    #               f"--repr_layers 33 " \
    #               f"--include per_tok " \
    #               f"--truncation_seq_length 10000 " \
    #               f"--model_dir ./esm_models"
    #         do(cmd)

    '''Predictions (the protein pose is also dynamic)'''
    model_workdir = f"./workdir/big_score_model_sanyueqi_with_time" # first download the model from https://zenodo.org/records/10137507
    ckpt = "ema_inference_epoch314_model.pt"

    for proteinName in list(protein_dict.keys()):

        ligandFile_with_protein_path = os.path.join(args.ligandCSVDirs, f"{proteinName}.csv")

        cmd = f"/home/ruofan/anaconda3/envs/diffdock/bin/python ./screening.py " \
                f"--seed {args.seed} " \
                f"--protein_dynamic" \
                f"--save_visualisation " \
                f"--model_dir {model_workdir}  " \
                f"--ckpt {ckpt} " \
                f"--protein_ligand_csv {ligandFile_with_protein_path} " \
                f"--esm_embeddings_path data/esm2_output_{proteinName} " \
                f"--out_dir {args.results}/{header}_{proteinName} " \
                f"--inference_steps {args.inference_steps} " \
                f"--samples_per_complex {args.samples_per_complex} " \
                f"--savings_per_complex {args.savings_per_complex} " \
                f"--batch_size 5 " \
                f"--actual_steps {args.inference_steps} " \
                f"--no_final_step_noise"

        do(cmd)

    print("hts complete.")

