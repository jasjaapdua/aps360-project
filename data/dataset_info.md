# Dataset Information

This project is dataset-agnostic and defaults to using the HuggingFace `spotify/podbaby-lyrics` dataset as an example. You can substitute any lyric dataset that exposes a `lyrics` text field.

## Expected columns
- `lyrics`: full song lyrics as a single string
- `artist` (optional): artist name for filtering or analysis
- `language` (optional): two-letter language code

## Preprocessing steps
1. Filter for English-language entries when `language` is provided.
2. Drop songs with fewer than `min_tokens` tokens after cleaning.
3. Remove section markers like `[chorus]`, `[verse 1]`.
4. Normalize to lowercase when `lowercase` is enabled.
5. Tokenize text, build vocabulary with special tokens `<pad>`, `<unk>`, `<bos>`, `<eos>`.
6. Split at the song level into train/validation/test.

## Updating the dataset
Edit `configs/dataset_config.yaml` to point to a new dataset and adjust preprocessing thresholds. Then run:

```bash
python scripts/download_datasets.py
python scripts/preprocess_data.py
```

Processed artifacts will be stored under `data/processed/`.
