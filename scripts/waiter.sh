#!/bin/bash

# Define the folder path you are waiting for
FOLDER_PATH="run_datagen/results/full"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Wait until the folder exists
while [ ! -d "$FOLDER_PATH" ]; do
  echo "Waiting for folder $FOLDER_PATH to exist..."
  sleep 5  # Wait for 5 seconds before checking again
done

echo "Folder $FOLDER_PATH found!"

# Run your series of commands here
echo "Running commands..."

# Example commands (replace these with your actual commands)
echo "Running preprocess_dpo.py..."
python scripts/preprocess_dpo.py || exit 1
echo "Running runner.sh..."
bash scripts/runner.sh || exit 1
echo "All commands executed."