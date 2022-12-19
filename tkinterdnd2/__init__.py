# dnd actions
PRIVATE = 'private'
NONE = 'none'
ASK = 'ask'
COPY = 'copy'
MOVE = 'move'
LINK = 'link'
REFUSE_DROP = 'refuse_drop'

# dnd types
DND_TEXT = 'DND_Text'
DND_FILES = 'DND_Files'
DND_ALL = '*'
CF_UNICODETEXT = 'CF_UNICODETEXT'
CF_TEXT = 'CF_TEXT'
CF_HDROP = 'CF_HDROP'

FileGroupDescriptor = 'FileGroupDescriptor - FileContents'# ??
FileGroupDescriptorW = 'FileGroupDescriptorW - FileContents'# ??

from . import TkinterDnD
from .TkinterDnD import Tk
from .TkinterDnD import TixTk


