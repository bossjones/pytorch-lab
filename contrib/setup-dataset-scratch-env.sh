#!/usr/bin/env bash
# Creates the local scratch directory tree for the twitter/facebook/tiktok classification
# dataset. Run once from the repo root before populating train/test splits manually.
# Idempotent — existing directories are left untouched.

echo "Setting up scratch/datasets/scratch/{train,test}/{tiktok,facebook,twitter} ..."
mkdir -pv scratch/datasets/scratch/{train,test}/{tiktok,facebook,twitter} || true
echo "Done. Classification dataset scratch tree is ready under scratch/datasets/scratch/."

