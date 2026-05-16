#!/usr/bin/env bash
# Creates the local scratch directory tree for the twitter/facebook/tiktok classification
# dataset. Run once from the repo root before populating train/test splits manually.
# Idempotent — existing directories are left untouched.

mkdir -pv scratch/datasets/scratch/{train,test}/{tiktok,facebook,twitter} || true
