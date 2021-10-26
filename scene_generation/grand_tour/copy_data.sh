bucket="s3://far-falcon-assets/grand_tour"
tmpfile="$(mktemp)"
dataset="${HOME}/grand_tour_dataset"
echo "tmpfile: $tmpfile"

DEBUG=0

# ls all slice directories, excluding hidden files and nerfstudio_models directory.
# s5cmd ls $bucket/*_nerfstudio/slices/slice_*/* \
s5cmd ls $bucket/*_nerfstudio/slices/slice_*/* \
  | awk '{print $NF}' \
  | grep -vE '(^|/)\.|/nerfstudio_models/|mesh_filled\.ply$|pcd\.ply$' \
  | awk -v bucket="${bucket}" -v prefix="${bucket}/" -v dataset="${dataset}" -v DEBUG="${DEBUG}" '
{
  raw=$0
  if (DEBUG) printf("DEBUG[in ]: %s\n", raw) > "/dev/stderr"

  # Normalize input: full URL vs bare key
  if (raw ~ /^s3:\/\//) {
    src = raw
    key = raw
    sub("^"prefix, "", key)
  } else {
    key = raw
    src = prefix key
  }

  if (DEBUG) {
    printf("DEBUG[key]: %s\n", key) > "/dev/stderr"
    printf("DEBUG[src]: %s\n", src) > "/dev/stderr"
  }

  # Expect: <scene>_nerfstudio/slices/<slice>/<rest>
  if (match(key, /^([^/]+)_nerfstudio\/slices\/([^/]+)\/(.*)$/, m)) {
    scene=m[1]; slice=m[2]; rest=m[3]

    dest = dataset "/" scene "/" slice "/" rest

    if (DEBUG) {
      printf("DEBUG[ok ]: scene=%s slice=%s rest=%s\n", scene, slice, rest) > "/dev/stderr"
      printf("DEBUG[dst]: %s\n", dest) > "/dev/stderr"
    }

    printf("cp %s %s\n", src, dest)
  } else {
    if (DEBUG) printf("DEBUG[skip]: no match for key=%s\n", key) > "/dev/stderr"
    next
  }
}
END {
  if (DEBUG) print "DEBUG[end]: awk finished." > "/dev/stderr"
}
' > "$tmpfile"

echo "Prepared $(wc -l < "$tmpfile") copy commands"
head -n 5 "$tmpfile" || true

# Execute in parallel
s5cmd run "$tmpfile"

rm -f "$tmpfile"