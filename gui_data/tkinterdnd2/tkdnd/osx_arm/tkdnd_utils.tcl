#
# tkdnd_utils.tcl --
#
#    This file implements some utility procedures that are used by the TkDND
#    package.
#
# This software is copyrighted by:
# George Petasis, National Centre for Scientific Research "Demokritos",
# Aghia Paraskevi, Athens, Greece.
# e-mail: petasis@iit.demokritos.gr
#
# The following terms apply to all files associated
# with the software unless explicitly disclaimed in individual files.
#
# The authors hereby grant permission to use, copy, modify, distribute,
# and license this software and its documentation for any purpose, provided
# that existing copyright notices are retained in all copies and that this
# notice is included verbatim in any distributions. No written agreement,
# license, or royalty fee is required for any of the authorized uses.
# Modifications to this software may be copyrighted by their authors
# and need not follow the licensing terms described here, provided that
# the new terms are clearly indicated on the first page of each file where
# they apply.
#
# IN NO EVENT SHALL THE AUTHORS OR DISTRIBUTORS BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
# ARISING OUT OF THE USE OF THIS SOFTWARE, ITS DOCUMENTATION, OR ANY
# DERIVATIVES THEREOF, EVEN IF THE AUTHORS HAVE BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# THE AUTHORS AND DISTRIBUTORS SPECIFICALLY DISCLAIM ANY WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.  THIS SOFTWARE
# IS PROVIDED ON AN "AS IS" BASIS, AND THE AUTHORS AND DISTRIBUTORS HAVE
# NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
# MODIFICATIONS.
#

package require tkdnd
namespace eval ::tkdnd {
  namespace eval utils {
  };# namespace ::tkdnd::utils
  namespace eval text {
    variable _drag_tag           tkdnd::drag::selection::tag
    variable _state              {}
    variable _drag_source_widget {}
    variable _drop_target_widget {}
    variable _now_dragging       0
  };# namespace ::tkdnd::text
};# namespace ::tkdnd

bind TkDND_Drag_Text1 <ButtonPress-1>   {tkdnd::text::_begin_drag clear  1 %W %s %X %Y %x %y}
bind TkDND_Drag_Text1 <B1-Motion>       {tkdnd::text::_begin_drag motion 1 %W %s %X %Y %x %y}
bind TkDND_Drag_Text1 <B1-Leave>        {tkdnd::text::_TextAutoScan %W %x %y}
bind TkDND_Drag_Text1 <ButtonRelease-1> {tkdnd::text::_begin_drag reset  1 %W %s %X %Y %x %y}
bind TkDND_Drag_Text2 <ButtonPress-2>   {tkdnd::text::_begin_drag clear  2 %W %s %X %Y %x %y}
bind TkDND_Drag_Text2 <B2-Motion>       {tkdnd::text::_begin_drag motion 2 %W %s %X %Y %x %y}
bind TkDND_Drag_Text2 <ButtonRelease-2> {tkdnd::text::_begin_drag reset  2 %W %s %X %Y %x %y}
bind TkDND_Drag_Text3 <ButtonPress-3>   {tkdnd::text::_begin_drag clear  3 %W %s %X %Y %x %y}
bind TkDND_Drag_Text3 <B3-Motion>       {tkdnd::text::_begin_drag motion 3 %W %s %X %Y %x %y}
bind TkDND_Drag_Text3 <ButtonRelease-3> {tkdnd::text::_begin_drag reset  3 %W %s %X %Y %x %y}

# ----------------------------------------------------------------------------
#  Command tkdnd::text::drag_source
# ----------------------------------------------------------------------------
proc ::tkdnd::text::drag_source { mode path { types DND_Text } { event 1 } { tagprefix TkDND_Drag_Text } { tag sel } } {
  switch -exact -- $mode {
    register {
      $path tag bind $tag <ButtonPress-${event}> \
        [list tkdnd::text::_begin_drag press ${event} %W %s %X %Y %x %y]
      ## Set a binding to the widget, to put selection as data...
      bind $path <<DragInitCmd>> \
        [list ::tkdnd::text::DragInitCmd $path %t $tag]
      ## Set a binding to the widget, to remove selection if action is move...
      bind $path <<DragEndCmd>> \
        [list ::tkdnd::text::DragEndCmd $path %A $tag]
    }
    unregister {
      $path tag bind $tag <ButtonPress-${event}> {}
      bind $path <<DragInitCmd>> {}
      bind $path <<DragEndCmd>>  {}
    }
  }
  ::tkdnd::drag_source $mode $path $types $event $tagprefix
};# ::tkdnd::text::drag_source

# ----------------------------------------------------------------------------
#  Command tkdnd::text::drop_target
# ----------------------------------------------------------------------------
proc ::tkdnd::text::drop_target { mode path { types DND_Text } } {
  switch -exact -- $mode {
    register {
      bind $path <<DropPosition>> \
        [list ::tkdnd::text::DropPosition $path %X %Y %A %a %m]
      bind $path <<Drop>> \
        [list ::tkdnd::text::Drop $path %D %X %Y %A %a %m]
    }
    unregister {
      bind $path <<DropEnter>>      {}
      bind $path <<DropPosition>>   {}
      bind $path <<DropLeave>>      {}
      bind $path <<Drop>>           {}
    }
  }
  ::tkdnd::drop_target $mode $path $types
};# ::tkdnd::text::drop_target

# ----------------------------------------------------------------------------
#  Command tkdnd::text::DragInitCmd
# ----------------------------------------------------------------------------
proc ::tkdnd::text::DragInitCmd { path { types DND_Text } { tag sel } { actions { copy move } } } {
  ## Save the selection indices...
  variable _drag_source_widget
  variable _drop_target_widget
  set _drag_source_widget $path
  set _drop_target_widget {}
  _save_selection $path $tag
  list $actions $types [$path get $tag.first $tag.last]
};# ::tkdnd::text::DragInitCmd

# ----------------------------------------------------------------------------
#  Command tkdnd::text::DragEndCmd
# ----------------------------------------------------------------------------
proc ::tkdnd::text::DragEndCmd { path action { tag sel } } {
  variable _drag_source_widget
  variable _drop_target_widget
  set _drag_source_widget {}
  set _drop_target_widget {}
  _restore_selection $path $tag
  switch -exact -- $action {
    move {
      ## Delete the original selected text...
      variable _selection_first
      variable _selection_last
      $path delete $_selection_first $_selection_last
    }
  }
};# ::tkdnd::text::DragEndCmd

# ----------------------------------------------------------------------------
#  Command tkdnd::text::DropPosition
# ----------------------------------------------------------------------------
proc ::tkdnd::text::DropPosition { path X Y action actions keys} {
  variable _drag_source_widget
  variable _drop_target_widget
  set _drop_target_widget $path
  ## This check is primitive, a more accurate one is needed!
  if {$path eq $_drag_source_widget} {
    ## This is a drag within the same widget! Set action to move...
    if {"move" in $actions} {set action move}
  }
  incr X -[winfo rootx $path]
  incr Y -[winfo rooty $path]
  $path mark set insert @$X,$Y; update
  return $action
};# ::tkdnd::text::DropPosition

# ----------------------------------------------------------------------------
#  Command tkdnd::text::Drop
# ----------------------------------------------------------------------------
proc ::tkdnd::text::Drop { path data X Y action actions keys } {
  incr X -[winfo rootx $path]
  incr Y -[winfo rooty $path]
  $path mark set insert @$X,$Y
  $path insert [$path index insert] $data
  return $action
};# ::tkdnd::text::Drop

# ----------------------------------------------------------------------------
#  Command tkdnd::text::_save_selection
# ----------------------------------------------------------------------------
proc ::tkdnd::text::_save_selection { path tag} {
  variable _drag_tag
  variable _selection_first
  variable _selection_last
  variable _selection_tag $tag
  set _selection_first [$path index $tag.first]
  set _selection_last  [$path index $tag.last]
  $path tag add $_drag_tag $_selection_first $_selection_last
  $path tag configure $_drag_tag \
    -background [$path tag cget $tag -background] \
    -foreground [$path tag cget $tag -foreground]
};# tkdnd::text::_save_selection

# ----------------------------------------------------------------------------
#  Command tkdnd::text::_restore_selection
# ----------------------------------------------------------------------------
proc ::tkdnd::text::_restore_selection { path tag} {
  variable _drag_tag
  variable _selection_first
  variable _selection_last
  $path tag delete $_drag_tag
  $path tag remove $tag 0.0 end
  #$path tag add $tag $_selection_first $_selection_last
};# tkdnd::text::_restore_selection

# ----------------------------------------------------------------------------
#  Command tkdnd::text::_begin_drag
# ----------------------------------------------------------------------------
proc ::tkdnd::text::_begin_drag { event button source state X Y x y } {
  variable _drop_target_widget
  variable _state
  # puts "::tkdnd::text::_begin_drag $event $button $source $state $X $Y $x $y"

  switch -exact -- $event {
    clear {
      switch -exact -- $_state {
         press {
           ## Do not execute other bindings, as they will erase selection...
           return -code break
         }
      }
      set _state clear
    }
    motion {
      variable _now_dragging
      if {$_now_dragging} {return -code break}
      if { [string equal $_state "press"] } {
        variable _x0; variable _y0
        if { abs($_x0-$X) > ${::tkdnd::_dx} || abs($_y0-$Y) > ${::tkdnd::_dy} } {
          set _state "done"
          set _drop_target_widget {}
          set _now_dragging       1
          set code [catch {
            ::tkdnd::_init_drag $button $source $state $X $Y $x $y
          } info options]
          set _drop_target_widget {}
          set _now_dragging       0
          if {$code != 0} {
            ## Something strange occurred...
            return -options $options $info
          }
        }
        return -code break
      }
      set _state clear
    }
    press {
      variable _x0; variable _y0
      set _x0    $X
      set _y0    $Y
      set _state "press"
    }
    reset {
      set _state {}
    }
  }
  if {$source eq $_drop_target_widget} {return -code break}
  return -code continue
};# tkdnd::text::_begin_drag

proc ::tkdnd::text::_TextAutoScan {w x y} {
  variable _now_dragging
  if {$_now_dragging} {return -code break}
  return -code continue
};# tkdnd::text::_TextAutoScan
