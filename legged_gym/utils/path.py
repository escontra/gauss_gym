import fnmatch
import glob as globlib
import io
import os
import re
import shutil
import threading
import time
import uuid


class Path:

  __slots__ = ('_path',)

  filesystems = []

  def __new__(cls, path):
    if cls is not Path:
      return super().__new__(cls)
    path = str(path)
    for impl, pred in cls.filesystems:
      if pred(path):
        obj = super().__new__(impl)
        obj.__init__(path)
        return obj
    raise NotImplementedError(f'No filesystem supports: {path}')

  def __getnewargs__(self):
    return (self._path,)

  def __init__(self, path):
    assert isinstance(path, str)
    path = re.sub(r'^\./*', '', path)  # Remove leading dot or dot slashes.
    path = re.sub(r'(?<=[^/])/$', '', path)  # Remove single trailing slash.
    path = path or '.'  # Empty path is represented by a dot.
    self._path = path

  def __truediv__(self, part):
    sep = '' if self._path.endswith('/') else '/'
    return type(self)(f'{self._path}{sep}{str(part)}')

  def __repr__(self):
    return f'Path({str(self)})'

  def __fspath__(self):
    return str(self)

  def __eq__(self, other):
    return self._path == other._path

  def __lt__(self, other):
    return self._path < other._path

  def __str__(self):
    return self._path

  def __hash__(self):
    return hash(str(self))

  @property
  def parent(self):
    if '/' not in self._path:
      return type(self)('.')
    parent = self._path.rsplit('/', 1)[0]
    parent = parent or ('/' if self._path.startswith('/') else '.')
    return type(self)(parent)

  @property
  def name(self):
    if '/' not in self._path:
      return self._path
    return self._path.rsplit('/', 1)[1]

  @property
  def stem(self):
    return self.name.split('.', 1)[0] if '.' in self.name else self.name

  @property
  def suffix(self):
    return ('.' + self.name.split('.', 1)[1]) if '.' in self.name else ''

  @property
  def size(self):
    raise NotImplementedError

  def read(self, mode='r'):
    assert mode in 'r rb'.split(), mode
    with self.open(mode) as f:
      return f.read()

  def read_text(self):
    return self.read(mode='r')

  def read_bytes(self):
    return self.read(mode='rb')

  def write(self, content, mode='w'):
    assert mode in 'w a wb ab'.split(), mode
    with self.open(mode) as f:
      f.write(content)

  def write_text(self, content):
    self.write(content, mode='w')

  def write_bytes(self, content):
    self.write(content, mode='wb')

  def open(self, mode='r'):
    raise NotImplementedError

  def absolute(self):
    raise NotImplementedError

  def glob(self, pattern):
    raise NotImplementedError

  def exists(self):
    raise NotImplementedError

  def isfile(self):
    raise NotImplementedError

  def isdir(self):
    raise NotImplementedError

  def mkdir(self, **kwargs):
    assert kwargs.pop('parents', True)
    assert kwargs.pop('exist_ok', True)
    assert not kwargs, kwargs
    raise NotImplementedError

  def remove(self, recursive=False):
    if recursive:
      for path in reversed(list(self.glob('**'))):
        path.remove()
    else:
      raise NotImplementedError

  def copy(self, dest, recursive=False):
    _copy_across_filesystems(self, dest, recursive)

  def move(self, dest, recursive=False):
    self.copy(dest, recursive)
    self.remove(recursive)


class LocalPath(Path):

  __slots__ = ('_path',)

  def __init__(self, path):
    super().__init__(os.path.expanduser(str(path)))

  @property
  def size(self):
    return os.path.getsize(str(self))

  def open(self, mode='r'):
    return open(str(self), mode=mode)

  def absolute(self):
    return type(self)(os.path.absolute(str(self)))

  def glob(self, pattern):
    for path in globlib.glob(f'{str(self)}/{pattern}', recursive=True):
      yield type(self)(path)

  def exists(self):
    return os.path.exists(str(self))

  def isfile(self):
    return os.path.isfile(str(self))

  def isdir(self):
    return os.path.isdir(str(self))

  def mkdir(self, **kwargs):
    assert kwargs.pop('parents', True)
    assert kwargs.pop('exist_ok', True)
    assert not kwargs, kwargs
    os.makedirs(str(self), exist_ok=True)

  def remove(self, recursive=False):
    if recursive:
      shutil.rmtree(self)
    else:
      if self.isdir():
        os.rmdir(str(self))
      else:
        os.remove(str(self))

  def copy(self, dest, recursive=False):
    dest = Path(dest)
    if isinstance(dest, type(self)):
      shutil.copy2(self, type(self)(dest))
    else:
      _copy_across_filesystems(self, dest, recursive)

  def move(self, dest, recursive=False):
    dest = Path(dest)
    if isinstance(dest, type(self)):
      shutil.move(self, dest)
    else:
      _copy_across_filesystems(self, dest, recursive)
      self.remove(recursive)


class TFPath(Path):

  __slots__ = ('_path',)

  gfile = None

  def __init__(self, path):
    path = str(path)
    if not (path.startswith('/') or '://' in path):
      path = os.path.abspath(os.path.expanduser(path))
    super().__init__(path)
    if not type(self).gfile:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
      type(self).gfile = tf.io.gfile

  @property
  def size(self):
    return self.gfile.stat(str(self)).st_size

  def open(self, mode='r'):
    path = str(self)
    if 'a' in mode and path.startswith('/cns/'):
      path += '%r=3.2'
    if mode.startswith('x'):
      if self.exists():
        raise FileExistsError(path)
      mode = mode.replace('x', 'w')
    return self.gfile.GFile(path, mode)

  def absolute(self):
    return self

  def glob(self, pattern):
    for path in self.gfile.glob(f'{str(self)}/{pattern}'):
      yield type(self)(path)

  def exists(self):
    return self.gfile.exists(str(self))

  def isfile(self):
    return self.exists() and not self.isdir()

  def isdir(self):
    return self.gfile.isdir(str(self))

  def mkdir(self, **kwargs):
    assert kwargs.pop('parents', True)
    assert kwargs.pop('exist_ok', True)
    assert not kwargs, kwargs
    self.gfile.makedirs(str(self))

  def remove(self, recursive=False):
    if recursive:
      self.gfile.rmtree(str(self))
    else:
      self.gfile.remove(str(self))

  def copy(self, dest, recursive=False):
    dest = Path(dest)
    if isinstance(dest, type(self)) and not recursive:
      self.gfile.copy(str(self), str(dest), overwrite=True)
    else:
      _copy_across_filesystems(self, dest, recursive)

  def move(self, dest, recursive=False):
    dest = Path(dest)
    if isinstance(dest, type(self)) and not recursive:
      self.gfile.rename(self, str(dest), overwrite=True)
    else:
      _copy_across_filesystems(self, dest, recursive)
      self.remove()


def gcs_retry(duration=60):
  from google.cloud.storage.retry import DEFAULT_RETRY
  return dict(timeout=duration, retry=DEFAULT_RETRY.with_deadline(duration))


GCS_LOCK = threading.RLock()
GCS_CLIENT = None
GCS_BUCKETS = {}


class GCSPath(Path):

  __slots__ = ('_path', '_blob')

  def __init__(self, path):
    path = str(path)
    super().__init__(path)
    self._blob = None

  def __getstate__(self):
    return {'path': self._path}

  def __setstate__(self, d):
    self._path = d['path']
    self._blob = None

  @property
  def blob(self):
    if not self._blob:
      from google.cloud import storage
      path = str(self)[5:]
      name = path.split('/', 1)[1] if '/' in path[5:] else None
      self._blob = name and storage.Blob(name, self.bucket)
    return self._blob

  @property
  def bucket(self):
    bucket = str(self)[5:].split('/', 1)[0]
    if bucket not in GCS_BUCKETS:
      with GCS_LOCK:
        if bucket not in GCS_BUCKETS:
          import google
          try:
            GCS_BUCKETS[bucket] = self.client.get_bucket(bucket, **gcs_retry())
          except google.api_core.exceptions.NotFound:
            return None
    return GCS_BUCKETS[bucket]

  @property
  def client(self):
    global GCS_CLIENT
    if not GCS_CLIENT:
      with GCS_LOCK:
        if not GCS_CLIENT:
          from google import auth
          from google.cloud import storage
          import requests
          credentials, project = auth.default()
          client = storage.Client(project, credentials)
          # https://github.com/googleapis/python-storage/issues/253
          adapter = requests.adapters.HTTPAdapter(
              pool_connections=64, pool_maxsize=64,
              max_retries=3, pool_block=True)
          client._http.mount('https://', adapter)
          client._http._auth_request.session.mount('https://', adapter)
          GCS_CLIENT = client
    return GCS_CLIENT

  @property
  def size(self):
    if self.blob.size is None:
      self._blob = self.bucket.get_blob(self.blob.name, **gcs_retry())
    assert isinstance(self._blob.size, int), self._blob.size
    return self._blob.size

  def open(self, mode='r'):
    assert self.blob, 'is a directory'
    if 'r' in mode:
      return GCSReadFile(self.blob, self.client)
    elif mode in ('a', 'ab'):
      return GCSAppendFile(self.blob, self.client, mode)
    else:
      # Supports writes as resumeable uploads.
      return self.blob.open(mode, ignore_flush=True, **gcs_retry(300))

  def read(self, mode='r'):
    assert self.blob, 'is a directory'
    if mode == 'rb':
      return self.blob.download_as_bytes(
          self.client, raw_download=True, **gcs_retry(300))
    elif mode == 'r':
      return self.read('rb').decode('utf-8')
    else:
      raise NotImplementedError(mode)

  def write(self, content, mode='w'):
    assert mode in 'w a wb ab'.split(), mode
    if mode == 'a':
      prefix = self.read('r')
      content = prefix + content
    if mode == 'ab':
      prefix = self.read('rb')
      content = prefix + content
    self.blob.upload_from_string(content, **gcs_retry(300))

  def absolute(self):
    return self

  def glob(self, pattern):
    pattern = pattern.rstrip('/')
    assert pattern
    prefix = self.blob.name + '/' if self.blob else ''
    if pattern == '*':
      response = self.bucket.list_blobs(prefix=prefix, delimiter='/')
      filenames = [x.name for x in response]  # Iterating also fills prefixes.
      folders = [x.rstrip('/') for x in response.prefixes]
    elif '**' in pattern:
      # Expand full depth. Look one level deeper to extract prefixes.
      pattern2 = prefix + pattern
      pattern2 += '' if pattern2.endswith('*') else '*'
      response = self.bucket.list_blobs(prefix=prefix, match_glob=pattern2)
      filenames = [x.name for x in response]
      folders = [x.rsplit('/', 1)[0] for x in filenames if '/' in x]
      folders = list(set(folders))
    else:
      # Glob for files and iteratively look for folders.
      pattern2 = prefix + pattern
      response = self.bucket.list_blobs(prefix=prefix, match_glob=pattern2)
      filenames = [x.name for x in response]
      parts = pattern.rstrip('/').split('/')
      queue = [(prefix, 0)]
      folders = []
      while queue:
        root, depth = queue.pop(0)
        part = parts[depth]
        if not any(x in part for x in '*[]{}'):  # Workaround for naive glob.
          response = self.bucket.list_blobs(prefix=root + part, delimiter='/')
        else:
          response = self.bucket.list_blobs(
              prefix=root, match_glob=root + part + '/', delimiter='/')
        results = list(response)  # Fetch all pages.
        results = list(response.prefixes)
        for child in results:
          if depth < len(parts) - 1:
            queue.append((child, depth + 1))
          else:
            folders.append(child)
    results = [x.rstrip('/') for x in folders + filenames]
    results = fnmatch.filter(results, prefix + pattern)
    results = sorted(set(results))
    return [type(self)(f'gs://{self.bucket.name}/{x}') for x in results]

  def exists(self):
    if not self.bucket:
      return False
    return self.isfile() or self.isdir()

  def isfile(self):
    if not self.bucket or not self.blob:
      return False
    return self.blob.exists(self.client)

  def isdir(self):
    from google.cloud import storage
    if not self.bucket:
      return False
    if not self.blob:
      return self.bucket.exists()
    if self.isfile():
      return False
    if storage.Blob(self.blob.name + '/', self.bucket).exists():
      return True
    try:
      next(iter(self.glob('*')))
      return True
    except StopIteration:
      return False

  def mkdir(self, **kwargs):
    assert kwargs.pop('parents', True)
    assert kwargs.pop('exist_ok', True)
    assert not kwargs, kwargs
    from google.cloud import storage
    if not self.blob:
      return
    if self.exists():
      return
    folder = storage.Blob(self.blob.name + '/', self.bucket)
    folder.upload_from_string(b'', **gcs_retry())

  def remove(self, recursive=False):
    from google.cloud import storage
    isdir = self.isdir()
    isfile = self.isfile()
    if recursive:
      assert not isfile
      for child in self.glob('**'):
        child.remove()
    if isdir:
      storage.Blob(self.blob.name + '/', self.bucket).delete(**gcs_retry())
    if isfile:
      self.blob.delete(self.client, **gcs_retry())

  def copy(self, dest, recursive=False):
    dest = Path(dest)
    if isinstance(dest, type(self)) and not recursive:
      self.bucket.copy_blob(
          self.blob, dest.bucket, dest.blob.name, **gcs_retry())
    else:
      _copy_across_filesystems(self, dest, recursive)

  def move(self, dest, recursive=False):
    dest = Path(dest)
    if isinstance(dest, type(self)) and not recursive:
      if self.bucket.name == dest.bucket.name:
        self.bucket.rename_blob(self.blob, dest.blob.name, **gcs_retry())
      else:
        self.bucket.copy_blob(
            self.blob, dest.bucket, dest.blob.name, **gcs_retry())
    else:
      _copy_across_filesystems(self, dest, recursive)
      self.remove()

  def _bucket(self, path):
    if not str(path).startswith('gs://'):
      return None
    return type(self)(path).bucket.name


class GCSReadFile:

  def __init__(self, blob, client):
    self.blob = blob
    self.client = client
    self.fetched = False
    self.pos = 0

  def __enter__(self):
    return self

  def __exit__(self, *e):
    self.close()

  def readable(self):
    return True

  def writeable(self):
    return False

  def seekable(self):
    return True

  def tell(self):
    return self.pos

  def seek(self, pos, mode=os.SEEK_SET):
    if not self.fetched:
      self.blob = self.blob.bucket.get_blob(self.blob.name, **gcs_retry())
      self.fetched = True
    if mode == os.SEEK_SET:
      self.pos = pos
    elif mode == os.SEEK_CUR:
      self.pos += pos
    elif mode == os.SEEK_END:
      self.pos = self.blob.size + pos
    else:
      raise NotImplementedError(mode)
    assert 0 <= self.pos <= self.blob.size, (self.pos, self.blob.size)

  def read(self, size=None):
    if size is None:
      buffer = self.blob.download_as_bytes(
          self.client, start=self.pos or None, raw_download=True,
          **gcs_retry())
      self.pos += len(buffer)
      return buffer
    if not self.fetched:
      self.blob = self.blob.bucket.get_blob(self.blob.name, **gcs_retry())
      self.fetched = True
    end = min(self.pos + size, self.blob.size)
    result = self.blob.download_as_bytes(
        self.client, start=self.pos, end=end, raw_download=True,
        **gcs_retry())
    assert 1 <= len(result) < size + 8, (
        self.blob.name, self.blob.size, self.pos, size, len(result))
    self.pos = end
    return result[:size]

  def close(self):
    pass


class GCSAppendFile:

  def __init__(self, blob, client, mode='a'):
    from google.cloud import storage
    self.client = client
    self.target = blob
    self.temp = storage.Blob('tmp/' + str(uuid.uuid4()), blob.bucket)
    self.fp = self.temp.open(mode.replace('a', 'w'), **gcs_retry(300))

  def __enter__(self):
    return self

  def __exit__(self, *e):
    self.close()

  def readable(self):
    return False

  def writeable(self):
    return True

  def seekable(self):
    return False

  def tell(self):
    raise io.UnsupportedOperation

  def seek(self, pos, mode=os.SEEK_SET):
    raise io.UnsupportedOperation

  def read(self, size=None):
    raise io.UnsupportedOperation

  def write(self, b):
    self.fp.write(b)

  def close(self):
    import google.cloud.exceptions
    self.fp.close()
    if not self.target.exists(**gcs_retry()):
      self.target.upload_from_string(b'', **gcs_retry())
    self._wait_until_exists(self.target)
    self._wait_until_exists(self.temp)
    self.target.compose([self.target, self.temp], **gcs_retry())
    try:
      self.temp.delete(**gcs_retry())
    except google.cloud.exceptions.NotFound:
      pass

  def _wait_until_exists(self, blob, timeout=60):
    start = time.time()
    while not blob.exists(**gcs_retry()):
      time.sleep(0.2)
      if time.time() - start >= timeout:
        raise TimeoutError


def _copy_across_filesystems(source, dest, recursive):
  assert isinstance(source, Path), type(source)
  assert isinstance(dest, Path), type(dest)
  if not recursive:
    assert source.isfile()
    dest.write(source.read('rb'), 'wb')
    return
  if dest.exists():
    dest = dest / source.name
  dest.mkdir()
  prefix = str(source)
  for s in source.glob('**'):
    assert str(s).startswith(prefix), (source, s)
    d = dest / str(s)[len(prefix):].lstrip('/')
    if s.isdir():
      d.mkdir()
    else:
      d.write(s.read('rb'), 'wb')


Path.filesystems = [
    (GCSPath, lambda path: path.startswith('gs://')),
    (TFPath, lambda path: path.startswith('/cns/')),
    (LocalPath, lambda path: True),
]