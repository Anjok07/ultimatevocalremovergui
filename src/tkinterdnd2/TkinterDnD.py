'''Python wrapper for the tkdnd tk extension.

The tkdnd extension provides an interface to native, platform specific
drag and drop mechanisms. Under Unix the drag & drop protocol in use is
the XDND protocol version 5 (also used by the Qt toolkit, and the KDE and
GNOME desktops). Under Windows, the OLE2 drag & drop interfaces are used.
Under Macintosh, the Cocoa drag and drop interfaces are used.

Once the TkinterDnD2 package is installed, it is safe to do:

from TkinterDnD2 import *

This will add the classes TkinterDnD.Tk and TkinterDnD.TixTk to the global
namespace, plus the following constants:
PRIVATE, NONE, ASK, COPY, MOVE, LINK, REFUSE_DROP,
DND_TEXT, DND_FILES, DND_ALL, CF_UNICODETEXT, CF_TEXT, CF_HDROP,
FileGroupDescriptor, FileGroupDescriptorW

Drag and drop for the application can then be enabled by using one of the
classes TkinterDnD.Tk() or (in case the tix extension shall be used)
TkinterDnD.TixTk() as application main window instead of a regular
tkinter.Tk() window. This will add the drag-and-drop specific methods to the
Tk window and all its descendants.
'''

try:
    import Tkinter as tkinter
    import Tix as tix
except ImportError:
    import tkinter
    from tkinter import tix

TkdndVersion = None

def _require(tkroot):
    '''Internal function.'''
    global TkdndVersion
    try:
        import os.path
        import platform

        if platform.system()=="Darwin":
            tkdnd_platform_rep = "osx64"
        elif platform.system()=="Linux":
            tkdnd_platform_rep = "linux64"
        elif platform.system()=="Windows":
            tkdnd_platform_rep = "win64"
        else:
            raise RuntimeError('Plaform not supported.')
        
        module_path = os.path.join(os.path.dirname(__file__), 'tkdnd', tkdnd_platform_rep)
        tkroot.tk.call('lappend', 'auto_path', module_path)
        TkdndVersion = tkroot.tk.call('package', 'require', 'tkdnd')
    except tkinter.TclError:
        raise RuntimeError('Unable to load tkdnd library.')
    return TkdndVersion

class DnDEvent:
    """Internal class.
    Container for the properties of a drag-and-drop event, similar to a
    normal tkinter.Event.
    An instance of the DnDEvent class has the following attributes:
        action (string)
        actions (tuple)
        button (int)
        code (string)
        codes (tuple)
        commonsourcetypes (tuple)
        commontargettypes (tuple)
        data (string)
        name (string)
        types (tuple)
        modifiers (tuple)
        supportedsourcetypes (tuple)
        sourcetypes (tuple)
        type (string)
        supportedtargettypes (tuple)
        widget (widget instance)
        x_root (int)
        y_root (int)
    Depending on the type of DnD event however, not all attributes may be set.
    """
    pass

class DnDWrapper:
    '''Internal class.'''
    # some of the percent substitutions need to be enclosed in braces
    # so we can use splitlist() to convert them into tuples
    _subst_format_dnd = ('%A', '%a', '%b', '%C', '%c', '{%CST}',
                         '{%CTT}', '%D', '%e', '{%L}', '{%m}', '{%ST}',
                         '%T', '{%t}', '{%TT}', '%W', '%X', '%Y')
    _subst_format_str_dnd = " ".join(_subst_format_dnd)
    tkinter.BaseWidget._subst_format_dnd = _subst_format_dnd
    tkinter.BaseWidget._subst_format_str_dnd = _subst_format_str_dnd

    def _substitute_dnd(self, *args):
        """Internal function."""
        if len(args) != len(self._subst_format_dnd):
            return args
        def getint_event(s):
            try:
                return int(s)
            except ValueError:
                return s
        def splitlist_event(s):
            try:
                return self.tk.splitlist(s)
            except ValueError:
                return s
        # valid percent substitutions for DnD event types
        # (tested with tkdnd-2.8 on debian jessie):
        # <<DragInitCmd>> : %W, %X, %Y %e, %t
        # <<DragEndCmd>> : %A, %W, %e
        # <<DropEnter>> : all except : %D (always empty)
        # <<DropLeave>> : all except %D (always empty)
        # <<DropPosition>> :all except %D (always empty)
        # <<Drop>> : all
        A, a, b, C, c, CST, CTT, D, e, L, m, ST, T, t, TT, W, X, Y = args
        ev = DnDEvent()
        ev.action = A
        ev.actions = splitlist_event(a)
        ev.button = getint_event(b)
        ev.code = C
        ev.codes = splitlist_event(c)
        ev.commonsourcetypes = splitlist_event(CST)
        ev.commontargettypes = splitlist_event(CTT)
        ev.data = D
        ev.name = e
        ev.types = splitlist_event(L)
        ev.modifiers = splitlist_event(m)
        ev.supportedsourcetypes = splitlist_event(ST)
        ev.sourcetypes = splitlist_event(t)
        ev.type = T
        ev.supportedtargettypes = splitlist_event(TT)
        try:
            ev.widget = self.nametowidget(W)
        except KeyError:
            ev.widget = W
        ev.x_root = getint_event(X)
        ev.y_root = getint_event(Y)
        return (ev,)
    tkinter.BaseWidget._substitute_dnd = _substitute_dnd

    def _dnd_bind(self, what, sequence, func, add, needcleanup=True):
        """Internal function."""
        if isinstance(func, str):
            self.tk.call(what + (sequence, func))
        elif func:
            funcid = self._register(func, self._substitute_dnd, needcleanup)
            # FIXME: why doesn't the "return 'break'" mechanism work here??
            #cmd = ('%sif {"[%s %s]" == "break"} break\n' % (add and '+' or '',
            #                              funcid, self._subst_format_str_dnd))
            cmd = '%s%s %s' %(add and '+' or '', funcid,
                                    self._subst_format_str_dnd)
            self.tk.call(what + (sequence, cmd))
            return funcid
        elif sequence:
            return self.tk.call(what + (sequence,))
        else:
            return self.tk.splitlist(self.tk.call(what))
    tkinter.BaseWidget._dnd_bind = _dnd_bind

    def dnd_bind(self, sequence=None, func=None, add=None):
        '''Bind to this widget at drag and drop event SEQUENCE a call
        to function FUNC.
        SEQUENCE may be one of the following:
        <<DropEnter>>, <<DropPosition>>, <<DropLeave>>, <<Drop>>,
        <<Drop:type>>, <<DragInitCmd>>, <<DragEndCmd>> .
        The callbacks for the <Drop*>> events, with the exception of
        <<DropLeave>>, should always return an action (i.e. one of COPY,
        MOVE, LINK, ASK or PRIVATE).
        The callback for the <<DragInitCmd>> event must return a tuple
        containing three elements: the drop action(s) supported by the
        drag source, the format type(s) that the data can be dropped as and
        finally the data that shall be dropped. Each of these three elements
        may be a tuple of strings or a single string.'''
        return self._dnd_bind(('bind', self._w), sequence, func, add)
    tkinter.BaseWidget.dnd_bind = dnd_bind

    def drag_source_register(self, button=None, *dndtypes):
        '''This command will register SELF as a drag source.
        A drag source is a widget than can start a drag action. This command
        can be executed multiple times on a widget.
        When SELF is registered as a drag source, optional DNDTYPES can be
        provided. These DNDTYPES will be provided during a drag action, and
        it can contain platform independent or platform specific types.
        Platform independent are DND_Text for dropping text portions and
        DND_Files for dropping a list of files (which can contain one or
        multiple files) on SELF. However, these types are
        indicative/informative. SELF can initiate a drag action with even a
        different type list. Finally, button is the mouse button that will be
        used for starting the drag action. It can have any of the values 1
        (left mouse button), 2 (middle mouse button - wheel) and 3
        (right mouse button). If button is not specified, it defaults to 1.'''
        # hack to fix a design bug from the first version
        if button is None:
            button = 1
        else:
            try:
                button = int(button)
            except ValueError:
                # no button defined, button is actually
                # something like DND_TEXT
                dndtypes = (button,) + dndtypes
                button = 1
        self.tk.call(
                'tkdnd::drag_source', 'register', self._w, dndtypes, button)
    tkinter.BaseWidget.drag_source_register = drag_source_register

    def drag_source_unregister(self):
        '''This command will stop SELF from being a drag source. Thus, window
        will stop receiving events related to drag operations. It is an error
        to use this command for a window that has not been registered as a
        drag source with drag_source_register().'''
        self.tk.call('tkdnd::drag_source', 'unregister', self._w)
    tkinter.BaseWidget.drag_source_unregister = drag_source_unregister

    def drop_target_register(self, *dndtypes):
        '''This command will register SELF as a drop target. A drop target is
        a widget than can accept a drop action. This command can be executed
        multiple times on a widget. When SELF is registered as a drop target,
        optional DNDTYPES can be provided. These types list can contain one or
        more types that SELF will accept during a drop action, and it can
        contain platform independent or platform specific types. Platform
        independent are DND_Text for dropping text portions and DND_Files for
        dropping a list of files (which can contain one or multiple files) on
        SELF.'''
        self.tk.call('tkdnd::drop_target', 'register', self._w, dndtypes)
    tkinter.BaseWidget.drop_target_register = drop_target_register

    def drop_target_unregister(self):
        '''This command will stop SELF from being a drop target. Thus, SELF
        will stop receiving events related to drop operations. It is an error
        to use this command for a window that has not been registered as a
        drop target with drop_target_register().'''
        self.tk.call('tkdnd::drop_target', 'unregister', self._w)
    tkinter.BaseWidget.drop_target_unregister = drop_target_unregister

    def platform_independent_types(self, *dndtypes):
        '''This command will accept a list of types that can contain platform
        independnent or platform specific types. A new list will be returned,
        where each platform specific type in DNDTYPES will be substituted by
        one or more platform independent types. Thus, the returned list may
        have more elements than DNDTYPES.'''
        return self.tk.split(self.tk.call(
                            'tkdnd::platform_independent_types', dndtypes))
    tkinter.BaseWidget.platform_independent_types = platform_independent_types

    def platform_specific_types(self, *dndtypes):
        '''This command will accept a list of types that can contain platform
        independnent or platform specific types. A new list will be returned,
        where each platform independent type in DNDTYPES will be substituted
        by one or more platform specific types. Thus, the returned list may
        have more elements than DNDTYPES.'''
        return self.tk.split(self.tk.call(
                            'tkdnd::platform_specific_types', dndtypes))
    tkinter.BaseWidget.platform_specific_types = platform_specific_types

    def get_dropfile_tempdir(self):
        '''This command will return the temporary directory used by TkDND for
        storing temporary files. When the package is loaded, this temporary
        directory will be initialised to a proper directory according to the
        operating system. This default initial value can be changed to be the
        value of the following environmental variables:
        TKDND_TEMP_DIR, TEMP, TMP.'''
        return self.tk.call('tkdnd::GetDropFileTempDirectory')
    tkinter.BaseWidget.get_dropfile_tempdir = get_dropfile_tempdir

    def set_dropfile_tempdir(self, tempdir):
        '''This command will change the temporary directory used by TkDND for
        storing temporary files to TEMPDIR.'''
        self.tk.call('tkdnd::SetDropFileTempDirectory', tempdir)
    tkinter.BaseWidget.set_dropfile_tempdir = set_dropfile_tempdir

#######################################################################
####      The main window classes that enable Drag & Drop for      ####
####      themselves and all their descendant widgets:             ####
#######################################################################

class Tk(tkinter.Tk, DnDWrapper):
    '''Creates a new instance of a tkinter.Tk() window; all methods of the
    DnDWrapper class apply to this window and all its descendants.'''
    def __init__(self, *args, **kw):
        tkinter.Tk.__init__(self, *args, **kw)
        self.TkdndVersion = _require(self)

class TixTk(tix.Tk, DnDWrapper):
    '''Creates a new instance of a tix.Tk() window; all methods of the
    DnDWrapper class apply to this window and all its descendants.'''
    def __init__(self, *args, **kw):
        tix.Tk.__init__(self, *args, **kw)
        self.TkdndVersion = _require(self)
