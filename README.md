# ACPSSI
ACPSSI (AntiCancer Peptide prediction by Sequential & Structural Information) is a dual-stream deep learning framework that combines sequential information (sequence) and structural information (SMILES) of peptides to perform binary prediction (ACP / non-ACP).

Project Summary

Model name: ACPSSI

Goal: Improve accuracy and generalizability in anticancer peptide prediction by combining amino acid sequence features with molecular representations (SMILES).

Approach: Dual-stream architecture (Sequence stream + Structure stream), with concatenated outputs passed to an MLP for classification.

Key Features

Utilizes ProtT5-XL-UniRef50 to extract sequence features (1024-dimensional per residue).

Sequence processing via BiLSTM, followed by a Transformer encoder and aggregated with mean pooling.

Structural features extracted from SMILES using ChemBERTa (pooler output — 768 dimensions). Final layers of ChemBERTa are selectively fine-tuned for ACP peptide data.

Concatenation of sequence and structure vectors, followed by multiple fully connected layers (MLP) with Dropout and BatchNorm.

Training with Binary Cross-Entropy, default optimizer AdamW, using Early Stopping and learning rate reduction on plateau.

Datasets (as in the paper)

Dataset 1 — AntiCP2.0 Alternate: 1940 samples — 970 positive (ACP) and 970 negative.

Dataset 2 — cACP-DeepGram: 1144 samples — 572 positive and 572 negative.

Both datasets are balanced and used for evaluating the generalizability of the model.

Preprocessing (Key Notes)

Unusual residues (U, Z, O, B, J) are replaced with the token 'X' for compatibility with ProtT5 tokenizer.

Each sequence is converted to its corresponding SMILES representation (conversion script in scripts/convert_to_smiles.py).

Fixed random seed and 5-fold cross-validation are applied for reproducibility.

Model Architecture (Summary)

Sequence stream:

Input: amino acid sequence

Tokenization → ProtT5-XL-UniRef50 → matrix (L, 1024)

BiLSTM → Transformer encoder → mean pooling → x_seq

Structure stream:

Input: SMILES

ChemBERTa → pooler output (768) → (optional fine-tuning of final layers) → x_smiles

Fusion & Classifier:

x_fusion = [ x_seq | x_smiles ]

MLP (ReLU → Dropout → BatchNorm) → binary output → softmax

Highlight Results (from paper)
Final model performance (full ACPSSI)

Dataset 1 (AntiCP2.0 Alternate)

Accuracy: 0.935

MCC: 0.874

Specificity: 0.979

Sensitivity: 0.891

Dataset 2 (cACP-DeepGram)

Accuracy: 0.953

MCC: 0.906

Specificity: 0.961

Sensitivity: 0.946

Ablation study results (table of reported values)
Model configuration	Accuracy	MCC	Specificity	Sensitivity
Sequence-only features	0.917	0.835	0.932	0.902
Structure-only features	0.716	0.435	0.773	0.659
Without Transformer in Subnetwork1	0.719	0.441	0.773	0.664
Without BiLSTM in Subnetwork1	0.931	0.865	0.984	0.876
Full ACPSSI (Sequence + Structure)	0.935	0.874	0.979	0.891

These results demonstrate that combining sequence and structure provides the greatest improvement compared to using either individually.
