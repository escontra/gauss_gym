from gauss_gym.utils import hf_utils
# from huggingface_hub import HfApi

# api = HfApi()
# files = [f for f in api.list_repo_files(repo_id="alescontrela/gauss_gym_data", repo_type='dataset')]
# print(f'Need to download {len(files)} files')

# Download only specific folder
try:
  attempt_dir = hf_utils.snapshot_download(
    repo_id='alescontrela/gauss_gym_data',
    wait_for_rank_zero_download=True,
    wait_for_rank_zero_timeout=3.0,
  )
except RuntimeError as e:
  print(f'Caught Runtime error! {e}')


# Download only specific folder
local_dir = hf_utils.snapshot_download(
  repo_id='alescontrela/gauss_gym_data',
)

# # Download only specific folder
# local_dir_redownload = hf_utils.snapshot_download(
#     repo_id="alescontrela/gauss_gym_data",
#     wait_for_rank_zero_download=True
# )

# assert local_dir == local_dir_redownload
