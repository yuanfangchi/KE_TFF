#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/countries_S3/"
vocab_dir="datasets/data_preprocessed/countries_S3/vocab"
total_iterations=300
path_length=3
hidden_size=2
embedding_size=2
batch_size=42
beta=0.1
Lambda=0.02
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/countries_s3/"
model_load_dir="nothing"
load_model=0
nell_evaluation=0
distributed_training=1
split_random=1
transferred_training=0