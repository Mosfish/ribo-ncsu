import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import dotenv
dotenv.load_dotenv(".env")

import os
import argparse
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch_cluster
from torch_geometric.utils import coalesce, to_undirected
from src.constants import RNA_NUCLEOTIDES, RNA_ATOMS, FILL_VALUE, PURINES, PYRIMIDINES # 确保导入所有常量
from typing import List # 确保导入 List (get_backbone_coords 需要)
import biotite
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd

from src.data.data_utils import pdb_to_tensor, get_c4p_coords
from src.data.clustering_utils import cluster_sequence_identity, cluster_structure_similarity
from src.constants import DATA_PATH

import warnings
warnings.filterwarnings("ignore", category=biotite.structure.error.IncompleteStructureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


keep_insertions = True
keep_pseudoknots = False

def _compute_graphormer_features(edge_index, num_nodes, max_path_len=10):
        """
        计算 spd_matrix (N, N) 和 shortest_path_edges (N, N, K)
        """
        K = max_path_len

        # 1. (node, node) -> edge_idx map
        # 我们需要处理 `to_undirected` 带来的双向边
        edge_map = {}
        for idx, (u, v) in enumerate(edge_index.T):
            u, v = u.item(), v.item()
            edge_map[(u, v)] = idx
        
        # 2. Scipy FW a) 距离 (spd_matrix) 和 b) 前驱节点
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
        dist_matrix, predecessors = shortest_path(
            csgraph=adj, 
            directed=False, # V4 使用 to_undirected, 所以这里是 False
            return_predecessors=True
        )
        
        # 3. Post-process spd_matrix
        dist_matrix[dist_matrix == np.inf] = -1
        spd_matrix = torch.from_numpy(dist_matrix).to(dtype=torch.long)
        
        # 4. Post-process shortest_path_edges (N, N, K)
        shortest_path_edges = torch.full((num_nodes, num_nodes, K), -1, dtype=torch.long)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j or predecessors[i, j] == -1:
                    continue
                
                path_edges = []
                curr_node = j
                while curr_node != i:
                    prev_node = predecessors[i, curr_node]
                    if prev_node == -1:
                        path_edges = [] # Broken path
                        break
                    
                    # 检查 (prev, curr) 还是 (curr, prev) 在我们的 edge_map 中
                    # 因为 `edge_index` 是双向的, `edge_map` 会包含两个方向
                    edge_idx = edge_map.get((prev_node, curr_node), -1)
                    
                    if edge_idx != -1:
                        path_edges.append(edge_idx)
                    else:
                        # 备用检查 (以防万一)
                        edge_idx_reverse = edge_map.get((curr_node, prev_node), -1)
                        if edge_idx_reverse != -1:
                            path_edges.append(edge_idx_reverse)
                        
                    curr_node = prev_node
                
                path_edges.reverse() # Path is now from i to j
                
                if len(path_edges) > 0 and len(path_edges) <= K:
                    shortest_path_edges[i, j, :len(path_edges)] = torch.tensor(path_edges, dtype=torch.long)
        
        return spd_matrix, shortest_path_edges

def get_backbone_coords(
        atom_tensor: torch.FloatTensor, 
        sequence: str,
        pyrimidine_bb_indices: List[int] = [
            RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N1") 
        ],
        purine_bb_indices: List[int] = [
            RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N9")
        ],
        fill_value: float = 1e-5
    ):
    """Returns tensor of backbone atom coordinates: ``(L x 3 x 3)``

    Note: One can easily change the backbone representation here by changing
          the indices of the atoms to include in the backbone. The running
          example in the docstrings uses a 3-bead coarse grained representation.

    :param atom_tensor: AtomTensor of shape ``(N_residues, N_atoms, 3)``
    :type atom_tensor: torch.FloatTensor
    :param pyrimidine_bb_indices: List of indices of ``[P, C4', N1]`` atoms (in
        order) for C and U nucleotides.
    :type pyrimidine_bb_indices: List[int]
    :param purine_bb_indices: List of indices of ``[P, C4', N9]`` atoms (in
        order) for A and G nucleotides. 
    :type purine_bb_indices: List[int]
    :param fill_value: Value to fill missing entries with. Defaults to ``1e-5``.
    :type fill_value: float
    """
    # check that sequence is str
    assert isinstance(sequence, str), "Sequence must be a string"

    # get indices of purine/pyrimidine bases in sequence
    purine_indices = [i for i, base in enumerate(sequence) if base in PURINES]
    pyrimidine_indices = [i for i, base in enumerate(sequence) if base in PYRIMIDINES]

    # create tensor of backbone atoms
    backbone_tensor = (
        torch.zeros((atom_tensor.shape[0], len(purine_bb_indices), 3)) + fill_value
    ).float()
    backbone_tensor[purine_indices] = atom_tensor[purine_indices][:, purine_bb_indices, :]
    backbone_tensor[pyrimidine_indices] = atom_tensor[pyrimidine_indices][:, pyrimidine_bb_indices, :]
    return backbone_tensor

if __name__ == "__main__":
    print("--- Running V5-Patched process_data.py ---")
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_name', dest='expt_name', default='process_data', type=str)
    parser.add_argument('--tags', nargs='+', dest='tags', default=[])
    parser.add_argument('--no_wandb', action="store_true")
    parser.add_argument('--config', type=str, default='configs/default.yaml')  # 这段是新加
    args, unknown = parser.parse_known_args()

    # Initialise wandb
    if args.no_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY"), 
            config=args.config, 
            name=args.expt_name, 
            mode='disabled'
        )
    else:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"), 
            entity=os.environ.get("WANDB_ENTITY"), 
            config=args.config, 
            name=args.expt_name, 
            tags=args.tags,
            mode='online'
        )
    TOP_K = 32
    MAX_PATH_LEN = 10
    
    # 定义bb_indices
    pyrimidine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N1")]
    purine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N9")]
    
    # 定义 letter_to_num (来自 featurizer.py)
    letter_to_num = {n: i for i, n in enumerate(RNA_NUCLEOTIDES)}
    letter_to_num["_"] = len(letter_to_num)
    ##########################
    # Load metadata csv files
    ##########################

    print("\nLoading non-redundant equivalence class table")
    eq_class_table = pd.read_csv(os.path.join(DATA_PATH, "nrlist_3.306_4.0A.csv"), names=["eq_class", "representative", "members"], dtype=str)
    eq_class_table.eq_class = eq_class_table.eq_class.apply(lambda x: x.split("_")[2].split(".")[0])

    id_to_eq_class = {}
    eq_class_to_ids = {}
    for i, row in tqdm(eq_class_table.iterrows(), total=len(eq_class_table)):
        ids_in_class = []
        for member in row["members"].split(","):
            _member = member.replace("|", "_")
            _chains = _member.split("+")
            if len(_chains) > 1:
                _member = _chains[0]
                for chain in _chains[1:]:
                    _member += f"-{chain.split('_')[2]}"

            id_to_eq_class[_member] = row["eq_class"]
            ids_in_class.append(_member)
        eq_class_to_ids[row["eq_class"]] = ids_in_class

    print("\nLoading RNAsolo table")
    rnasolo_table = pd.read_csv(os.path.join(DATA_PATH, "rnasolo-main-table.csv"), dtype=str)
    rnasolo_table.eq_class = rnasolo_table.eq_class.apply(lambda x: str(x).split(".")[0])

    eq_class_to_type = {}
    for i, row in tqdm(rnasolo_table.iterrows(), total=len(rnasolo_table)):
        eq_class_to_type[row["eq_class"]] = row["molecule"]

    print("\nLoading RFAM table")
    rfam_table = pd.read_csv(os.path.join(DATA_PATH, "RFAM_families_27062023.csv"), dtype=str)

    id_to_rfam = {}
    for i, row in tqdm(rfam_table.iterrows(), total=len(rfam_table)):
        if row["pdb_id"].upper() not in id_to_rfam.keys():
            id_to_rfam[row["pdb_id"].upper()] = row["id"]

    ########################
    # Process raw PDB files
    ########################

    # Initialise empty dictionaries
    id_to_seq = {}
    seq_to_data = {}
    error_ids = []
    
    print(f"\nProcessing raw PDB files from {DATA_PATH}")
    filenames = tqdm(os.listdir(os.path.join(DATA_PATH, "raw")))
    for filename in filenames:
        try:
            structure_id, file_ext = os.path.splitext(filename)
            filenames.set_description(structure_id)
            if file_ext != ".pdb": continue

            # 1. 加载 PDB，假设 'coords' 是 (L, 27, 3) PyTorch 张量
            sequence, coords, sec_struct, sasa = pdb_to_tensor(
                os.path.join(DATA_PATH, "raw", filename),
                keep_insertions=keep_insertions,
                keep_pseudoknots=keep_pseudoknots
            )

            if len(sequence) <= 10: 
                continue

            # 2. 获取元数据
            rfam = id_to_rfam.get(structure_id.split("_")[0], "unknown")
            eq_class = id_to_eq_class.get(structure_id, "unknown")
            struct_type = eq_class_to_type.get(eq_class, "unknown")

            # 4. 检查是否为新序列
            if sequence in seq_to_data.keys():
                # --- A. 如果是重复序列 ---
                
                # 'coords' 是张量。 'coords_0' 是从字典中取出的 NumPy 数组。
                coords_0_np = seq_to_data[sequence]['coords_list'][0] # (L, 27, 3) NumPy
                coords_np = coords.cpu().numpy() # 将新加载的张量转为 NumPy

                # MDAnalysis 函数现在接收它们期望的 NumPy 数组
                R_hat = rotation_matrix(
                    get_c4p_coords(coords_np),  # <-- 传入 NumPy
                    get_c4p_coords(coords_0_np) # <-- 传入 NumPy
                )[0]
                
                coords_np = coords_np @ R_hat.T # 对齐 NumPy
                
                for other_id, other_coords_np in zip(seq_to_data[sequence]['id_list'], seq_to_data[sequence]['coords_list']):
                    # 'other_coords_np' 也已经是 NumPy
                    seq_to_data[sequence]['rmsds_list'][(structure_id, other_id)] = get_rmsd(
                        get_c4p_coords(coords_np),       # <-- 传入 NumPy
                        get_c4p_coords(other_coords_np), # <-- 传入 NumPy
                        superposition=True
                    )
                
                # append 对齐后的 (L, 27, 3) NumPy 数组
                seq_to_data[sequence]['id_list'].append(structure_id)
                seq_to_data[sequence]['coords_list'].append(coords_np) # 存入 NumPy
                seq_to_data[sequence]['sec_struct_list'].append(sec_struct)
                seq_to_data[sequence]['sasa_list'].append(sasa)
                seq_to_data[sequence]['rfam_list'].append(rfam)
                seq_to_data[sequence]['eq_class_list'].append(eq_class)
                seq_to_data[sequence]['type_list'].append(struct_type)
            
            else:
                # --- B. 如果是新序列！V5 的计算逻辑放在这里 ---
                
                # a. 'coords' 已经是张量，直接传递
                _coords_bb = get_backbone_coords(
                    coords.float(), # <-- 直接使用张量
                    sequence,
                    pyrimidine_bb_indices,
                    purine_bb_indices,
                    fill_value=FILL_VALUE
                )
                
                if torch.all((_coords_bb == FILL_VALUE).sum(axis=(1,2)) > 0):
                    continue 

                # b. 获取静态掩码 (使用 *原始* coords 张量)
                seq_tensor = torch.as_tensor([letter_to_num.get(r, 4) for r in sequence])
                mask_coords_static = (coords == FILL_VALUE).sum(axis=(1,2)) == 0
                mask_coords_static = (mask_coords_static) & (seq_tensor.to(coords.device) != 4)
                
                num_nodes = mask_coords_static.sum().item()
                if num_nodes <= 10:
                    continue 
                    
                # c. 构建静态 edge_index (使用掩码后的 backbone coords)
                _coords_bb_masked = _coords_bb[mask_coords_static] 
                edge_index = torch_cluster.knn_graph(
                    _coords_bb_masked.mean(1), 
                    TOP_K
                )
                edge_index = to_undirected(coalesce(edge_index))

                # d. 计算昂贵的 V5 特征
                print(f"Computing Graphormer features for graph {structure_id} with {num_nodes} nodes...")
                spd_matrix, shortest_path_edges = _compute_graphormer_features(
                    edge_index, num_nodes, max_path_len=MAX_PATH_LEN
                )
                print("    Finished.")

                # e. 创建新条目并保存 (L, 27, 3) NumPy coords
                seq_to_data[sequence] = {
                    'sequence': sequence,
                    'id_list': [structure_id],
                    'coords_list': [coords.cpu().numpy()], # <-- 转换为 NumPy 存储
                    'sec_struct_list': [sec_struct],
                    'sasa_list': [sasa],
                    'rfam_list': [rfam],
                    'eq_class_list': [eq_class],
                    'type_list': [struct_type],
                    'rmsds_list': {},
                    'cluster_seqid0.8': -1,
                    'cluster_structsim0.45': -1,

                    # --- V5 新增键 ---
                    'mask_coords_static': mask_coords_static, 
                    'edge_index': edge_index,
                    'spd_matrix': spd_matrix,
                    'shortest_path_edges': shortest_path_edges
                }
            
            id_to_seq[structure_id] = sequence
        
        # catch errors and check manually later
        except Exception as e:
            print(structure_id, e)
            error_ids.append((structure_id, e))

    print(f"\nSaving (partially) processed data to {DATA_PATH}")
    torch.save(seq_to_data, os.path.join(DATA_PATH, "processed.pt"))
    
    #########################################
    # Cluster sequences by sequence identity
    #########################################

    print("\nClustering at 80\% sequence similarity (CD-HIT-EST)")
    id_to_cluster_seqid = cluster_sequence_identity(
        [SeqRecord(Seq(seq), id=data["id_list"][0]) for seq, data in seq_to_data.items()],
        identity_threshold = 0.8,
    )
    if not id_to_cluster_seqid:
        print("WARNING: 'id_to_cluster_seqid' is empty. Skipping sequence clustering.")
        print("         This might be because 'seq_to_data' is empty or 'cd-hit-est' (external tool) failed to run.")
    else:
        unclustered_idx = max(list(id_to_cluster_seqid.values())) + 1
        for seq, data in seq_to_data.items():
            id = data["id_list"][0]
            if id in id_to_cluster_seqid.keys():
                seq_to_data[seq]['cluster_seqid0.8'] = id_to_cluster_seqid[id]
            else:
                seq_to_data[seq]['cluster_seqid0.8'] = unclustered_idx
                unclustered_idx += 1
    
    print(f"\nSaving (partially) processed data to {DATA_PATH}")
    torch.save(seq_to_data, os.path.join(DATA_PATH, "processed.pt"))

    #############################################
    # Cluster structures by structure similarity
    #############################################

    print("\nClustering at 45% structure similarity (US-align)")
    # using first structure per sequence
    cluster_list_structsim = cluster_structure_similarity(
        [os.path.join(DATA_PATH, "raw", data["id_list"][0]+".pdb") for data in seq_to_data.values()],
        similarity_threshold = 0.45
    )
    if not cluster_list_structsim:
        print("WARNING: 'cluster_list_structsim' is empty. Skipping structural clustering.")
        print("         This might be because 'US-align' (external tool) failed to run.")
    else:
        for i, cluster in enumerate(cluster_list_structsim):
            # 添加一个额外的检查，以防 id_to_seq 查找失败
                if id.split(".")[0] in id_to_seq:
                    seq_to_data[id_to_seq[id.split(".")[0]]]['cluster_structsim0.45'] = i
    # --- 修复结束 ---

    print(f"\nSaving processed data to {DATA_PATH}")
    torch.save(seq_to_data, os.path.join(DATA_PATH, "processed.pt"))

    # Save processed metadata to csv
    df = pd.DataFrame.from_dict(seq_to_data, orient="index", columns=["id_list", 'rfam_list', 'eq_class_list', 'type_list', 'cluster_seqid0.8', 'cluster_structsim0.45'])
    df["sequence"] = df.index
    df.reset_index(drop=True, inplace=True)
    df["length"] = df.sequence.apply(lambda x: len(x))
    df["mean_rmsd"] = df.sequence.apply(lambda x: np.mean(list(seq_to_data[x]["rmsds_list"].values()))).fillna(0.0)
    df["median_rmsd"] = df.sequence.apply(lambda x: np.median(list(seq_to_data[x]["rmsds_list"].values()))).fillna(0.0)
    df["num_structures"] = df.id_list.apply(lambda x: len(x))
    df.to_csv(os.path.join(DATA_PATH, "processed_df.csv"), index=False)

    # Print IDs with errors
    if len(error_ids) > 0:
        print("\nIDs with errors (check manually):")
        for id, error in error_ids:
            print(f"{id}: {error}")

