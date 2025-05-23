import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from tqdm import tqdm
import ast # For safely evaluating string representations of lists
import os # For checking if best model file exists

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
EMBEDDING_L2_REG = 1e-4 
BATCH_SIZE = 2048
NUM_EPOCHS = 50 # Adjust as needed (early stopping will be added)
EMBEDDING_DIM = 64
NUM_LAYERS = 3 
TOP_K_EVAL = 10 # For validation/test recall
TOP_K_SUBMISSION = 10 # For the final submission file

# --- File Paths (Update these to your actual file paths) ---
TRAIN_FILE = 'train_formatted.csv'
VALID_FILE = 'valid_metrics_format.csv'
TEST_FILE = 'test_metrics_format.csv' # Used for metrics, not submission users
SAMPLE_SUBMISSION_FILE = 'sample_submission.csv' # Defines users for submission
SUBMISSION_FILE_NAME = 'submission.csv'
BEST_MODEL_PATH = "best_lightgcn_model.pth"


# --- 1. Data Preprocessing ---
# (load_and_preprocess_data, _extract_raw_ids_from_neg_items_for_map_building, parse_and_map_neg_items functions remain the same as your provided correct version)
# This parser is used when mapping with the final item_map
def parse_and_map_neg_items(neg_str_or_list, item_map_dict):
    """Safely parses string representation of list and maps item IDs to indices."""
    try:
        if isinstance(neg_str_or_list, str):
            if neg_str_or_list.startswith('[') and neg_str_or_list.endswith(']'):
                neg_list_orig_ids = ast.literal_eval(neg_str_or_list)
            else: 
                neg_list_orig_ids = [int(i.strip()) for i in neg_str_or_list.split(',')]
        elif isinstance(neg_str_or_list, list): 
            neg_list_orig_ids = neg_str_or_list
        else: 
            neg_list_orig_ids = [int(i.strip()) for i in str(neg_str_or_list).split(',')]
        
        return [item_map_dict[i] for i in neg_list_orig_ids if i in item_map_dict]
    except Exception:
        return []

# This helper is used ONLY for collecting all raw item IDs before item_map is created
def _extract_raw_ids_from_neg_items_for_map_building(neg_str_or_list):
    try:
        ids_to_return = []
        if isinstance(neg_str_or_list, str):
            if neg_str_or_list.startswith('[') and neg_str_or_list.endswith(']'):
                ids_to_return = ast.literal_eval(neg_str_or_list)
            else:
                ids_to_return = [s.strip() for s in neg_str_or_list.split(',')]
        elif isinstance(neg_str_or_list, list):
            ids_to_return = neg_str_or_list
        else: 
            ids_to_return = [s.strip() for s in str(neg_str_or_list).split(',')]
        # Ensure all elements are at least strings for pd.to_numeric later
        return [str(item) for item in ids_to_return]
    except Exception:
        return []

def load_and_preprocess_data(train_file_path, valid_file_path, test_file_path=None, sample_submission_file_path=None):
    print("Loading data...")
    train_df, valid_df, test_df, sample_sub_df = None, None, None, None
    try:
        train_df = pd.read_csv(train_file_path)
        valid_df = pd.read_csv(valid_file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Critical file missing.")
        
    if test_file_path:
        try:
            test_df = pd.read_csv(test_file_path)
        except FileNotFoundError:
            print(f"Warning: Test file '{test_file_path}' not found. Skipping test set.")
            test_df = None
    
    if sample_submission_file_path:
        try:
            sample_sub_df = pd.read_csv(sample_submission_file_path)
        except FileNotFoundError:
            print(f"Warning: Sample submission file '{sample_submission_file_path}' not found.")
            # If you need sample_sub_df for ID mapping, this could be an issue later.
            # For now, we'll proceed, assuming user_ids for mapping might come from other DFs.
            sample_sub_df = None


    # Consolidate all user IDs
    user_id_source_dfs = [df for df in [train_df, valid_df, test_df, sample_sub_df] if df is not None and 'user_id' in df.columns]

    if not user_id_source_dfs: raise ValueError("No dataframes with 'user_id' column loaded.")
    
    all_user_ids_series = pd.concat([df['user_id'] for df in user_id_source_dfs])
    all_user_ids_np = all_user_ids_series.dropna().unique() # NumPy array

    # Consolidate all item IDs
    item_id_source_series_list = []
    # Add items from train/valid/test item_id and neg_items
    for df_source in [df for df in [train_df, valid_df, test_df] if df is not None]: 
        if 'item_id' in df_source.columns:
            item_id_source_series_list.append(df_source['item_id'])
        if 'neg_items' in df_source.columns:
            item_id_source_series_list.append(
                df_source['neg_items'].apply(_extract_raw_ids_from_neg_items_for_map_building).explode()
            )
    # Add items from sample_submission 'item_id' if it contains example items (it usually does for format)
    if sample_sub_df is not None and 'item_id' in sample_sub_df.columns:
         item_id_source_series_list.append(
                sample_sub_df['item_id'].apply(_extract_raw_ids_from_neg_items_for_map_building).explode()
            )

    if not item_id_source_series_list:
        all_item_ids_np = np.array([], dtype=int)
    else:
        master_item_id_series = pd.concat(item_id_source_series_list)
        master_item_id_series = master_item_id_series.dropna() 
        master_item_id_series_numeric = pd.to_numeric(master_item_id_series, errors='coerce')
        master_item_id_series_numeric_no_na = master_item_id_series_numeric.dropna()
        master_item_id_series_int = master_item_id_series_numeric_no_na.astype(int)
        all_item_ids_np = master_item_id_series_int.unique()

    user_map = {id_val: i for i, id_val in enumerate(all_user_ids_np)}
    item_map = {id_val: i for i, id_val in enumerate(all_item_ids_np)}
    num_users = len(user_map)
    num_items = len(item_map)

    print(f"Num users: {num_users}, Num items: {num_items}")
    if num_users == 0 or num_items == 0:
        raise ValueError("No users or items found after mapping. Check data integrity and parsing.")

    # Map IDs to indices in all DataFrames
    if train_df is not None:
        train_df['user_idx'] = train_df['user_id'].map(user_map)
        train_df['item_idx'] = train_df['item_id'].map(item_map)
        train_df.dropna(subset=['user_idx', 'item_idx'], inplace=True)
        train_df['user_idx'] = train_df['user_idx'].astype(int)
        train_df['item_idx'] = train_df['item_idx'].astype(int)

    if valid_df is not None:
        valid_df['user_idx'] = valid_df['user_id'].map(user_map)
        valid_df['item_idx'] = valid_df['item_id'].map(item_map)
        if 'neg_items' in valid_df.columns:
            valid_df['neg_item_indices'] = valid_df['neg_items'].apply(lambda x: parse_and_map_neg_items(x, item_map))
        else:
            valid_df['neg_item_indices'] = pd.Series([[] for _ in range(len(valid_df))])

    if test_df is not None:
        test_df['user_idx'] = test_df['user_id'].map(user_map)
        test_df['item_idx'] = test_df['item_id'].map(item_map)
        if 'neg_items' in test_df.columns:
            test_df['neg_item_indices'] = test_df['neg_items'].apply(lambda x: parse_and_map_neg_items(x, item_map))
        else:
            test_df['neg_item_indices'] = pd.Series([[] for _ in range(len(test_df))])
    
    if train_df is None or train_df.empty:
        raise ValueError("Training data is empty or None after processing. Cannot build graph.")

    rows = train_df['user_idx'].values
    cols = train_df['item_idx'].values
    data = np.ones(len(rows))
    R = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R_dok = R.todok()
    for i, j in zip(*R_dok.nonzero()):
        adj_mat[i, j + num_users] = 1
        adj_mat[j + num_users, i] = 1 
    adj_mat = adj_mat.tocoo()

    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()

    values = norm_adj_mat.data
    indices = np.vstack((norm_adj_mat.row, norm_adj_mat.col))
    i_torch = torch.LongTensor(indices)
    v_torch = torch.FloatTensor(values)
    shape = norm_adj_mat.shape
    graph_tensor = torch.sparse.FloatTensor(i_torch, v_torch, torch.Size(shape)).to(DEVICE)

    return train_df, valid_df, test_df, sample_sub_df, num_users, num_items, graph_tensor, user_map, item_map


# --- 2. LightGCN Model --- (Identical to your provided version)
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, num_layers, graph):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.graph = graph 

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def computer(self):
        users_emb_0 = self.user_embedding.weight
        items_emb_0 = self.item_embedding.weight
        all_emb_0 = torch.cat([users_emb_0, items_emb_0])
        
        embs = [all_emb_0] 
        current_emb = all_emb_0
        for _ in range(self.num_layers):
            current_emb = torch.sparse.mm(self.graph, current_emb)
            embs.append(current_emb)

        final_embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(final_embs, dim=1)
        
        final_users_emb, final_items_emb = torch.split(final_embs, [self.num_users, self.num_items])
        return final_users_emb, final_items_emb, users_emb_0, items_emb_0

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        final_users_emb, final_items_emb, initial_users_emb, initial_items_emb = self.computer()

        batch_final_user_embs = final_users_emb[user_indices]
        batch_final_pos_item_embs = final_items_emb[pos_item_indices]
        batch_final_neg_item_embs = final_items_emb[neg_item_indices]

        pos_scores = torch.sum(batch_final_user_embs * batch_final_pos_item_embs, dim=1)
        neg_scores = torch.sum(batch_final_user_embs * batch_final_neg_item_embs, dim=1)

        batch_initial_user_embs = initial_users_emb[user_indices]
        batch_initial_pos_item_embs = initial_items_emb[pos_item_indices]
        batch_initial_neg_item_embs = initial_items_emb[neg_item_indices]
        
        return pos_scores, neg_scores, batch_initial_user_embs, batch_initial_pos_item_embs, batch_initial_neg_item_embs

    def get_user_ratings(self, user_indices_tensor):
        final_users_emb, final_items_emb, _, _ = self.computer()
        batch_user_final_embs = final_users_emb[user_indices_tensor]
        ratings = torch.matmul(batch_user_final_embs, final_items_emb.t())
        return ratings

# --- 3. BPR Loss --- (Identical)
def bpr_loss_fn(pos_scores, neg_scores, initial_user_emb, initial_pos_item_emb, initial_neg_item_emb, l2_reg_weight):
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    reg_loss = (initial_user_emb.norm(2).pow(2) +
                initial_pos_item_emb.norm(2).pow(2) +
                initial_neg_item_emb.norm(2).pow(2)) / float(len(pos_scores))
    return loss + l2_reg_weight * reg_loss

# --- 4. Training Loop --- (Identical)
def train_epoch(model, optimizer, train_data_df, num_all_items, b_size, l2_reg):
    model.train()
    total_loss = 0.0
    if train_data_df is None or train_data_df.empty:
        print("Warning: Training data is empty. Skipping epoch.")
        return 0.0
        
    user_item_pairs = train_data_df[['user_idx', 'item_idx']].values.astype(int)
    np.random.shuffle(user_item_pairs) 

    user_pos_items = train_data_df.groupby('user_idx')['item_idx'].apply(set).to_dict()

    if len(user_item_pairs) == 0:
        print("Warning: No user-item pairs for training after processing.")
        return 0.0

    num_batches = 0
    for i in tqdm(range(0, len(user_item_pairs), b_size), desc="Training Batches"):
        num_batches += 1
        batch_pairs = user_item_pairs[i:i+b_size]
        user_indices = torch.LongTensor(batch_pairs[:, 0]).to(DEVICE)
        pos_item_indices = torch.LongTensor(batch_pairs[:, 1]).to(DEVICE)

        neg_item_indices_list = []
        for user_idx_val in user_indices.cpu().numpy():
            user_pos_set = user_pos_items.get(user_idx_val, set())
            neg_item = np.random.randint(0, num_all_items)
            tries = 0
            while neg_item in user_pos_set and tries < 100 : 
                neg_item = np.random.randint(0, num_all_items)
                tries +=1
            neg_item_indices_list.append(neg_item)
        neg_item_indices = torch.LongTensor(neg_item_indices_list).to(DEVICE)

        optimizer.zero_grad()
        pos_scores, neg_scores, initial_u_emb, initial_pos_i_emb, initial_neg_i_emb = \
            model(user_indices, pos_item_indices, neg_item_indices)
        
        loss = bpr_loss_fn(pos_scores, neg_scores, initial_u_emb, initial_pos_i_emb, initial_neg_i_emb, l2_reg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / num_batches if num_batches > 0 else 0.0

# --- 5. Evaluation --- (Identical)
@torch.no_grad()
def evaluate_recall_at_k(model_eval, eval_df, top_k_val, user_map_dict, item_map_dict):
    model_eval.eval()
    num_hits = 0
    total_samples = 0

    if eval_df is None or eval_df.empty:
        # print(f"Warning: Evaluation DataFrame is empty. Skipping Recall@{top_k_val}.")
        return 0.0

    final_users_emb_eval, final_items_emb_eval, _, _ = model_eval.computer()

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Evaluating Recall@{top_k_val}"):
        user_original_id = row['user_id']
        pos_item_original_id = row['item_id']
        
        user_idx = user_map_dict.get(user_original_id)
        pos_item_idx = item_map_dict.get(pos_item_original_id)
        
        neg_item_indices_mapped = row.get('neg_item_indices', [])

        if user_idx is None or pos_item_idx is None: continue
        if not isinstance(neg_item_indices_mapped, list) or not neg_item_indices_mapped: continue

        user_idx_tensor = torch.LongTensor([user_idx]).to(DEVICE)
        
        candidate_item_pool_indices = [pos_item_idx] + neg_item_indices_mapped
        candidate_item_pool_indices = [idx for idx in candidate_item_pool_indices if idx is not None] 
        
        if not candidate_item_pool_indices or pos_item_idx not in candidate_item_pool_indices: continue

        candidate_items_tensor = torch.LongTensor(list(set(candidate_item_pool_indices))).to(DEVICE)
        
        try:
            pos_item_index_in_candidates = (candidate_items_tensor == pos_item_idx).nonzero(as_tuple=True)[0].item()
        except IndexError: 
            continue

        user_final_emb = final_users_emb_eval[user_idx_tensor] 
        candidate_final_item_embs = final_items_emb_eval[candidate_items_tensor] 

        scores = torch.matmul(user_final_emb, candidate_final_item_embs.t()).squeeze() 

        if scores.dim() == 0: scores = scores.unsqueeze(0)
        if len(scores) == 0: continue
        
        current_k = min(top_k_val, len(scores))
        if current_k == 0: continue

        _, top_ranked_indices_in_candidates = torch.topk(scores, current_k)

        if pos_item_index_in_candidates in top_ranked_indices_in_candidates.cpu().numpy():
            num_hits += 1
        total_samples += 1
    
    if total_samples == 0:
        return 0.0
    return num_hits / total_samples

# --- 6. Submission Generation ---
def generate_submission_file(model_to_submit,
                             target_user_ids_original,
                             user_map_dict,
                             item_map_dict,
                             n_total_items,
                             k_submission,
                             train_interactions_df, # Pass train_df here
                             submission_filename):
    model_to_submit.eval()
    item_rev_map = {v: k for k, v in item_map_dict.items()}
    
    # Get training interactions to exclude them from recommendations
    user_train_pos_items_original = {}
    if train_interactions_df is not None:
        # Use original IDs for this lookup, easier for filtering later
        user_train_pos_items_original = train_interactions_df.groupby('user_id')['item_id'].apply(set).to_dict()

    # Pre-compute global popular items (original IDs) for cold start/padding
    if train_interactions_df is not None and not train_interactions_df.empty:
        global_pop_items_series = train_interactions_df['item_id'].value_counts().index.tolist()
    else: # Fallback if no training data for popularity
        global_pop_items_series = [item_rev_map[i] for i in range(min(k_submission + 20, n_total_items)) if i in item_rev_map]


    predictions_list = []
    with torch.no_grad():
        final_users_emb_all, final_items_emb_all, _, _ = model_to_submit.computer()

        for original_user_id in tqdm(target_user_ids_original, desc="Generating Submission"):
            user_seen_items_original = user_train_pos_items_original.get(original_user_id, set())
            recommended_original_item_ids = []

            if original_user_id in user_map_dict:
                user_idx = user_map_dict[original_user_id]
                user_idx_tensor = torch.LongTensor([user_idx]).to(DEVICE)
                
                user_final_emb = final_users_emb_all[user_idx_tensor]
                ratings = torch.matmul(user_final_emb, final_items_emb_all.t()).squeeze()

                # Get more than K items initially to allow for filtering
                num_candidates = k_submission + len(user_seen_items_original) + 20 # Fetch more candidates
                if num_candidates > n_total_items : num_candidates = n_total_items
                
                if num_candidates > 0 and len(ratings) > 0: # Ensure ratings tensor is not empty
                    # Handle case where ratings might be a scalar if n_total_items = 1
                    if ratings.dim() == 0: ratings = ratings.unsqueeze(0)
                    
                    actual_num_candidates = min(num_candidates, len(ratings))
                    if actual_num_candidates > 0:
                        _, top_candidate_indices = torch.topk(ratings, k=actual_num_candidates)
                        top_candidate_indices = top_candidate_indices.cpu().numpy()

                        for item_idx in top_candidate_indices:
                            if item_idx < n_total_items and item_idx in item_rev_map:
                                original_item_id = item_rev_map[item_idx]
                                if original_item_id not in user_seen_items_original:
                                    recommended_original_item_ids.append(original_item_id)
                                if len(recommended_original_item_ids) == k_submission:
                                    break
                    else: # No candidates from topk (e.g., ratings len was 0)
                        pass # Will be handled by padding
                else: # No items to rank or ratings tensor empty
                    pass # Will be handled by padding

            

            item_id_string = ",".join(map(str, recommended_original_item_ids))
            predictions_list.append({
                "ID": original_user_id, # As per instruction: ID is user_id
                "user_id": original_user_id,
                "item_id": item_id_string
            })

    submission_df = pd.DataFrame(predictions_list)
    
    # Ensure order matches sample_submission.csv if it has a specific order of IDs
    try:
        sample_sub_df_for_order = pd.read_csv(SAMPLE_SUBMISSION_FILE)
        # Merge to maintain the order of IDs from sample_submission.csv
        # Use 'ID' for merging as it's guaranteed to be the user_id by problem spec
        submission_df_ordered = pd.merge(sample_sub_df_for_order[['ID']], submission_df, on='ID', how='left')
        # Fill user_id if it became NaN after left merge (should not happen if all target_user_ids were processed)
        submission_df_ordered['user_id'] = submission_df_ordered['user_id'].fillna(submission_df_ordered['ID'])
        # Fill item_id for any user in sample_submission not processed (e.g., extreme cold start not in user_map)
        # with a default empty or popular list string
        default_padding_string = ",".join(map(str, global_pop_items_series[:k_submission]))
        submission_df_ordered['item_id'] = submission_df_ordered['item_id'].fillna(default_padding_string)
        submission_df_ordered = submission_df_ordered[['ID', 'user_id', 'item_id']] # Ensure correct columns and order
    except FileNotFoundError:
        print(f"Warning: {SAMPLE_SUBMISSION_FILE} not found for ordering. Saving submission as is.")
        submission_df_ordered = submission_df # Use the unordered one

    submission_df_ordered.to_csv(submission_filename, index=False)
    print(f"{submission_filename} generated.")


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Include SAMPLE_SUBMISSION_FILE in loading
    train_df, valid_df, test_df, sample_submission_df_loaded, n_users, n_items, graph, user_map, item_map = \
        load_and_preprocess_data(TRAIN_FILE, VALID_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE)

    if n_users == 0 or n_items == 0:
        print("Error: No users or items after preprocessing. Exiting.")
        exit()
        
    model = LightGCN(n_users, n_items, EMBEDDING_DIM, NUM_LAYERS, graph).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    best_valid_recall = 0.0
    best_epoch = 0
    patience_epochs = 10 # Example: stop after 10 epochs of no improvement on validation
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_epoch(model, optimizer, train_df, n_items, BATCH_SIZE, EMBEDDING_L2_REG)
        print(f"Epoch {epoch}/{NUM_EPOCHS}, Avg. BPR Loss: {avg_loss:.4f}")

        current_valid_recall = 0.0
        if valid_df is not None and 'neg_item_indices' in valid_df.columns:
            current_valid_recall = evaluate_recall_at_k(model, valid_df, TOP_K_EVAL, user_map, item_map)
            print(f"Epoch {epoch}/{NUM_EPOCHS}, Validation Recall@{TOP_K_EVAL}: {current_valid_recall:.4f}")
            
            if current_valid_recall > best_valid_recall:
                best_valid_recall = current_valid_recall
                best_epoch = epoch
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"Saved new best model from epoch {best_epoch} with Recall@{TOP_K_EVAL}: {best_valid_recall:.4f}")
                patience_counter = 0 
            else:
                patience_counter += 1
            
            if patience_counter >= patience_epochs:
                print(f"Early stopping triggered at epoch {epoch}. Best recall {best_valid_recall:.4f} at epoch {best_epoch}.")
                break 
        else:
            print(f"Epoch {epoch}/{NUM_EPOCHS}, Skipping validation recall.")
            # If no validation, save model at last epoch or periodically
            if epoch == NUM_EPOCHS:
                 torch.save(model.state_dict(), BEST_MODEL_PATH)
                 print(f"Saved model from last epoch {epoch}.")


    print("Training finished.")
    
    # Load the best model for final evaluations and submission
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH} (Epoch {best_epoch}, Recall@{TOP_K_EVAL}: {best_valid_recall:.4f})")
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
    else:
        print("Warning: No best model saved. Using model from last epoch.")

    if valid_df is not None and 'neg_item_indices' in valid_df.columns:
        final_valid_recall = evaluate_recall_at_k(model, valid_df, TOP_K_EVAL, user_map, item_map)
        print(f"\n--- Final Validation Set Performance (using best/last model) ---")
        print(f"Validation Recall@{TOP_K_EVAL}: {final_valid_recall:.4f}")

    if test_df is not None and 'neg_item_indices' in test_df.columns:
        print("\n--- Final Test Set Performance (using best/last model) ---")
        test_recall = evaluate_recall_at_k(model, test_df, TOP_K_EVAL, user_map, item_map)
        print(f"Test Recall@{TOP_K_EVAL}: {test_recall:.4f}")
    else:
        print("\nNo test set or 'neg_item_indices' in test set. Skipping final test evaluation.")

    # --- Generate Submission File ---
    if sample_submission_df_loaded is not None and 'user_id' in sample_submission_df_loaded.columns:
        print("\n--- Generating Submission File ---")
        target_users_for_submission = sample_submission_df_loaded['user_id'].unique()
        generate_submission_file(model,
                                 target_users_for_submission,
                                 user_map,
                                 item_map,
                                 n_items,
                                 TOP_K_SUBMISSION,
                                 train_df, # Pass train_df for filtering seen items
                                 SUBMISSION_FILE_NAME)
    else:
        print("\nSample submission file not loaded or 'user_id' column missing. Cannot generate submission file.")