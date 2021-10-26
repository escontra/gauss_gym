from gauss_gym.utils import scene_ingest

# Test GaussGymData
# print('=' * 80)
# print('Testing GaussGymData (HuggingFace)')
# print('=' * 80)

# gg_data = scene_ingest.GaussGymData()
# valid_scenes = gg_data.find_valid_scene_directories()
# print(valid_scenes)

# gg_data.download_scenes(max_num_scenes=15)

# meshes = gg_data.load_meshes()

# mesh_0 = meshes[0]
# print(mesh_0.scene_name)
# print(mesh_0.filename)
# print(mesh_0.filepath)
# print(mesh_0.splatpath)
# print(mesh_0.vertices.shape)
# print(mesh_0.triangles.shape)
# print(mesh_0.cam_trans.shape)
# print(mesh_0.cam_offset.shape)
# print(mesh_0.ig_to_orig_rot.shape)
# print(mesh_0.cam_quat_xyzw.shape)

# Test GrandTourData (S3)
print('\n' + '=' * 80)
print('Testing GrandTourData (S3)')
print('=' * 80)

# Option 1: Use known scene paths (FAST - recommended for most use cases)
print('\n--- Using known scene paths (fast) ---')
known_paths = ['2024-10-01-11-47-44_nerfstudio/slices/slice_1148c086']
gt_data = scene_ingest.GrandTourData(known_scene_paths=known_paths)
print(f'Using known scenes: {gt_data._valid_scenes}')

# Option 2: Auto-discover scenes (SLOW - only use when needed)
# NOTE: Auto-discovery is very slow (~5+ minutes) due to large S3 bucket size.
# Uncomment below to test auto-discovery:
# print('\n--- Auto-discovering scenes (slow) ---')
# gt_data = scene_ingest.GrandTourData()
# valid_scenes = gt_data.find_valid_scene_directories(max_scenes_to_find=1)
# print(f'Found scenes: {valid_scenes}')

gt_data.download_scenes(max_num_scenes=1)

meshes = gt_data.load_meshes()

if meshes:
  mesh_0 = meshes[0]
  print(f'\nLoaded mesh successfully:')
  print(f'  Scene name: {mesh_0.scene_name}')
  print(f'  Filename: {mesh_0.filename}')
  print(f'  Filepath: {mesh_0.filepath}')
  print(f'  Splat path: {mesh_0.splatpath}')
  print(f'  Vertices shape: {mesh_0.vertices.shape}')
  print(f'  Triangles shape: {mesh_0.triangles.shape}')
  print(f'  Camera transforms shape: {mesh_0.cam_trans.shape}')
  print(f'  Camera offset shape: {mesh_0.cam_offset.shape}')
  print(f'  IG to orig rotation shape: {mesh_0.ig_to_orig_rot.shape}')
  print(f'  Camera quaternions shape: {mesh_0.cam_quat_xyzw.shape}')
