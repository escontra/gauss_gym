import re
import sys


REGEX_TOKEN = re.compile(
    r"([^a-zA-Z0-9-_./'\"\[\]]|['\"][^\s]*['\"])", re.MULTILINE)
REGEX_NUMBER = re.compile(
    r'([-+]?[0-9]+[0-9.,]*(e[-+]?[0-9])?|nan|-?inf)')
KEYWORDS = (
    'True', 'False', 'None', 'bool', 'int', 'str', 'float',
    'uint8', 'float16', 'float32', 'int32', 'int64')

escseq = lambda parts: '\033[' + ';'.join(parts) + 'm'
colors = dict(
    black=0, red=1, green=2, yellow=3, blue=4, magenta=5, cyan=6, white=7)


def style(color=None, background=None, bold=None, underline=None, reset=None):
  if not sys.stdout.isatty():
    return ''
  parts = []
  if reset:
    parts.append(escseq('0'))
  if color or bold or underline:
    args = ['3' + (str(colors[color]) if color else '9')]
    bold and args.append('1')
    underline and args.append('4')
    parts.append(escseq(args))
  if background:
    parts.append(escseq('4' + str(colors[background])))
  return ''.join(parts)


def print_(*values, color=True, bold=None, **kwargs):
  values = [format_(x) for x in values]
  value = kwargs.get('sep', ' ').join(str(x) for x in values)
  assert not color or isinstance(color, (bool, str)), color
  if (isinstance(color, str) or bold):
    args = []
    if isinstance(color, str) or bold:
      args.append(style(color=color, bold=bold))
    args.append(str(value))
    if isinstance(color, str) or bold:
      args.append(style(reset=True))
    value = ''.join(args)
  elif color is True:
    result = []
    prev = [None, None, None, None]  # Color, bold, underline, highlighted
    tokens = REGEX_TOKEN.split(value) + [None]
    for i, token in enumerate(tokens[:-1]):
      new = prev.copy()
      word = token.strip()
      new[1] = None
      if not word:
        new[0] = None
      elif word in '/-+':
        new[0] = 'green'
        new[1] = True
      elif word in '{}()<>,:':
        new[0] = 'white'
      elif token == '=':
        new[0] = 'white'
      elif word[0].isalpha() and tokens[i + 1] == '=':
        new[0] = 'magenta'
      elif word in KEYWORDS:
        new[0] = 'blue'
      elif word.startswith('---'):
        new[3] = True
      elif REGEX_NUMBER.match(word):
        new[0] = 'blue'
      elif word[0] == word[-1] == "'":
        new[0] = 'red'
      elif word[0] == word[-1] == '"':
        new[0] = 'red'
      elif word[0] == '[' and word[-1] == ']':
        new[0] = 'cyan'
      elif any(word.startswith(x) for x in ('/', '~', './')):
        new[0] = 'yellow'
      elif len(word) >= 3 and word[0] == word[-1] and word[0] in ("'", '"'):
        new[0] = 'green'
      elif word[0] == word[0].upper():
        new[0] = None
      else:
        new[0] = None
      if new[3]:  # Highlighted
        new[0] = 'cyan'
        new[1] = True
        new[2] = False
      if new != prev:
        result.append(style(
            color=new[0],
            bold=new[1],
            underline=new[2],
            reset=True))
      result.append(token)
      prev = new
      if '\n' in token:
        prev[1] = None
        prev[3] = None
    result.append(style(reset=True))
    value = ''.join(result)
  print(value, **kwargs)


def format_(value):
  if isinstance(value, dict):
    items = [f'{format_(k)}: {format_(v)}' for k, v in value.items()]
    return '{' + ', '.join(items) + '}'
  if isinstance(value, list):
    return '[' + ', '.join(f'{format_(x)}' for x in value) + ']'
  if isinstance(value, tuple):
    return '(' + ', '.join(f'{format_(x)}' for x in value) + ')'
  if all(hasattr(value, n) for n in ('shape', 'dtype', 'reshape')):
    shape = ','.join(str(x) for x in value.shape)
    dtype = value.dtype.name
    for long, short in {'float': 'f', 'uint': 'u', 'int': 'i'}.items():
      dtype = dtype.replace(long, short)
    return f'{dtype}<{shape}>'
  if isinstance(value, bytes):
    value = '0x' + value.hex() if r'\x' in str(value) else str(value)
    if len(value) > 32:
      value = value[:32 - 3] + '...'
  return str(value)