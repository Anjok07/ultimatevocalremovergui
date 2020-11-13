#
# tkdnd_generic.tcl --
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

namespace eval generic {
  variable _types {}
  variable _typelist {}
  variable _codelist {}
  variable _actionlist {}
  variable _pressedkeys {}
  variable _action {}
  variable _common_drag_source_types {}
  variable _common_drop_target_types {}
  variable _drag_source {}
  variable _drop_target {}

  variable _last_mouse_root_x 0
  variable _last_mouse_root_y 0

  variable _tkdnd2platform
  variable _platform2tkdnd

  proc debug {msg} {
    puts $msg
  };# debug

  proc initialise { } {
  };# initialise

  proc initialise_platform_to_tkdnd_types { types } {
    variable _platform2tkdnd
    variable _tkdnd2platform
    set _platform2tkdnd [dict create {*}$types]
    set _tkdnd2platform [dict create]
    foreach type [dict keys $_platform2tkdnd] {
      dict lappend _tkdnd2platform [dict get $_platform2tkdnd $type] $type
    }
  };# initialise_platform_to_tkdnd_types

  proc initialise_tkdnd_to_platform_types { types } {
    variable _tkdnd2platform
    set _tkdnd2platform [dict create {*}$types]
  };# initialise_tkdnd_to_platform_types

};# namespace generic

# ----------------------------------------------------------------------------
#  Command generic::HandleEnter
# ----------------------------------------------------------------------------
proc generic::HandleEnter { drop_target drag_source typelist codelist
                            actionlist pressedkeys } {
  variable _typelist;                 set _typelist    $typelist
  variable _pressedkeys;              set _pressedkeys $pressedkeys
  variable _action;                   set _action      refuse_drop
  variable _common_drag_source_types; set _common_drag_source_types {}
  variable _common_drop_target_types; set _common_drop_target_types {}
  variable _actionlist
  variable _drag_source;              set _drag_source $drag_source
  variable _drop_target;              set _drop_target {}
  variable _actionlist;               set _actionlist  $actionlist
  variable _codelist                  set _codelist    $codelist

  variable _last_mouse_root_x;        set _last_mouse_root_x 0
  variable _last_mouse_root_y;        set _last_mouse_root_y 0
  # debug "\n==============================================================="
  # debug "generic::HandleEnter: drop_target=$drop_target,\
  #        drag_source=$drag_source,\
  #        typelist=$typelist"
  # debug "generic::HandleEnter: ACTION: default"
  return default
};# generic::HandleEnter

# ----------------------------------------------------------------------------
#  Command generic::HandlePosition
# ----------------------------------------------------------------------------
proc generic::HandlePosition { drop_target drag_source pressedkeys
                               rootX rootY { time 0 } } {
  variable _types
  variable _typelist
  variable _codelist
  variable _actionlist
  variable _pressedkeys
  variable _action
  variable _common_drag_source_types
  variable _common_drop_target_types
  variable _drag_source
  variable _drop_target

  variable _last_mouse_root_x;        set _last_mouse_root_x $rootX
  variable _last_mouse_root_y;        set _last_mouse_root_y $rootY

  # debug "generic::HandlePosition: drop_target=$drop_target,\
  #            _drop_target=$_drop_target, rootX=$rootX, rootY=$rootY"

  if {![info exists _drag_source] && ![string length $_drag_source]} {
    # debug "generic::HandlePosition: no or empty _drag_source:\
    #               return refuse_drop"
    return refuse_drop
  }

  if {$drag_source ne "" && $drag_source ne $_drag_source} {
    debug "generic position event from unexpected source: $_drag_source\
           != $drag_source"
    return refuse_drop
  }

  set _pressedkeys $pressedkeys

  ## Does the new drop target support any of our new types?
  # foreach {common_drag_source_types common_drop_target_types} \
  #         [GetWindowCommonTypes $drop_target $_typelist] {break}
  foreach {drop_target common_drag_source_types common_drop_target_types} \
          [FindWindowWithCommonTypes $drop_target $_typelist] {break}
  set data [GetDroppedData $time]

  # debug "\t($_drop_target) -> ($drop_target)"
  if {$drop_target != $_drop_target} {
    if {[string length $_drop_target]} {
      ## Call the <<DropLeave>> event.
      # debug "\t<<DropLeave>> on $_drop_target"
      set cmd [bind $_drop_target <<DropLeave>>]
      if {[string length $cmd]} {
        set cmd [string map [list %W $_drop_target %X $rootX %Y $rootY \
          %CST \{$_common_drag_source_types\} \
          %CTT \{$_common_drop_target_types\} \
          %CPT \{[lindex [platform_independent_type [lindex $_common_drag_source_types 0]] 0]\} \
          %ST  \{$_typelist\}    %TT \{$_types\} \
          %A   \{$_action\}      %a \{$_actionlist\} \
          %b   \{$_pressedkeys\} %m \{$_pressedkeys\} \
          %D   \{\}              %e <<DropLeave>> \
          %L   \{$_typelist\}    %% % \
          %t   \{$_typelist\}    %T  \{[lindex $_common_drag_source_types 0]\} \
          %c   \{$_codelist\}    %C  \{[lindex $_codelist 0]\} \
          ] $cmd]
        uplevel \#0 $cmd
      }
    }
    set _drop_target $drop_target
    set _action      refuse_drop

    if {[llength $common_drag_source_types]} {
      set _action [lindex $_actionlist 0]
      set _common_drag_source_types $common_drag_source_types
      set _common_drop_target_types $common_drop_target_types
      ## Drop target supports at least one type. Send a <<DropEnter>>.
      # puts "<<DropEnter>> -> $drop_target"
      set cmd [bind $drop_target <<DropEnter>>]
      if {[string length $cmd]} {
        focus $drop_target
        set cmd [string map [list %W $drop_target %X $rootX %Y $rootY \
          %CST \{$_common_drag_source_types\} \
          %CTT \{$_common_drop_target_types\} \
          %CPT \{[lindex [platform_independent_type [lindex $_common_drag_source_types 0]] 0]\} \
          %ST  \{$_typelist\}    %TT \{$_types\} \
          %A   $_action          %a  \{$_actionlist\} \
          %b   \{$_pressedkeys\} %m  \{$_pressedkeys\} \
          %D   [list $data]      %e  <<DropEnter>> \
          %L   \{$_typelist\}    %%  % \
          %t   \{$_typelist\}    %T  \{[lindex $_common_drag_source_types 0]\} \
          %c   \{$_codelist\}    %C  \{[lindex $_codelist 0]\} \
          ] $cmd]
        set _action [uplevel \#0 $cmd]
        switch -exact -- $_action {
          copy - move - link - ask - private - refuse_drop - default {}
          default {set _action copy}
        }
      }
    }
  }

  set _drop_target {}
  if {[llength $common_drag_source_types]} {
    set _common_drag_source_types $common_drag_source_types
    set _common_drop_target_types $common_drop_target_types
    set _drop_target $drop_target
    ## Drop target supports at least one type. Send a <<DropPosition>>.
    set cmd [bind $drop_target <<DropPosition>>]
    if {[string length $cmd]} {
      set cmd [string map [list %W $drop_target %X $rootX %Y $rootY \
        %CST \{$_common_drag_source_types\} \
        %CTT \{$_common_drop_target_types\} \
        %CPT \{[lindex [platform_independent_type [lindex $_common_drag_source_types 0]] 0]\} \
        %ST  \{$_typelist\}    %TT \{$_types\} \
        %A   $_action          %a  \{$_actionlist\} \
        %b   \{$_pressedkeys\} %m  \{$_pressedkeys\} \
        %D   [list $data]      %e  <<DropPosition>> \
        %L   \{$_typelist\}    %%  % \
        %t   \{$_typelist\}    %T  \{[lindex $_common_drag_source_types 0]\} \
        %c   \{$_codelist\}    %C  \{[lindex $_codelist 0]\} \
        ] $cmd]
      set _action [uplevel \#0 $cmd]
    }
  }
  # Return values: copy, move, link, ask, private, refuse_drop, default
  # debug "generic::HandlePosition: ACTION: $_action"
  switch -exact -- $_action {
    copy - move - link - ask - private - refuse_drop - default {}
    default {set _action copy}
  }
  return $_action
};# generic::HandlePosition

# ----------------------------------------------------------------------------
#  Command generic::HandleLeave
# ----------------------------------------------------------------------------
proc generic::HandleLeave { } {
  variable _types
  variable _typelist
  variable _codelist
  variable _actionlist
  variable _pressedkeys
  variable _action
  variable _common_drag_source_types
  variable _common_drop_target_types
  variable _drag_source
  variable _drop_target
  variable _last_mouse_root_x
  variable _last_mouse_root_y
  if {![info exists _drop_target]} {set _drop_target {}}
  # debug "generic::HandleLeave: _drop_target=$_drop_target"
  if {[info exists _drop_target] && [string length $_drop_target]} {
    set cmd [bind $_drop_target <<DropLeave>>]
    if {[string length $cmd]} {
      set cmd [string map [list %W $_drop_target \
        %X $_last_mouse_root_x %Y $_last_mouse_root_y \
        %CST \{$_common_drag_source_types\} \
        %CTT \{$_common_drop_target_types\} \
        %CPT \{[lindex [platform_independent_type [lindex $_common_drag_source_types 0]] 0]\} \
        %ST  \{$_typelist\}    %TT \{$_types\} \
        %A   \{$_action\}      %a  \{$_actionlist\} \
        %b   \{$_pressedkeys\} %m  \{$_pressedkeys\} \
        %D   \{\}              %e  <<DropLeave>> \
        %L   \{$_typelist\}    %%  % \
        %t   \{$_typelist\}    %T  \{[lindex $_common_drag_source_types 0]\} \
        %c   \{$_codelist\}    %C  \{[lindex $_codelist 0]\} \
        ] $cmd]
      set _action [uplevel \#0 $cmd]
    }
  }
  foreach var {_types _typelist _actionlist _pressedkeys _action
               _common_drag_source_types _common_drop_target_types
               _drag_source _drop_target} {
    set $var {}
  }
};# generic::HandleLeave

# ----------------------------------------------------------------------------
#  Command generic::HandleDrop
# ----------------------------------------------------------------------------
proc generic::HandleDrop {drop_target drag_source pressedkeys rootX rootY time } {
  variable _types
  variable _typelist
  variable _codelist
  variable _actionlist
  variable _pressedkeys
  variable _action
  variable _common_drag_source_types
  variable _common_drop_target_types
  variable _drag_source
  variable _drop_target
  variable _last_mouse_root_x
  variable _last_mouse_root_y
  variable _last_mouse_root_x;        set _last_mouse_root_x $rootX
  variable _last_mouse_root_y;        set _last_mouse_root_y $rootY

  set _pressedkeys $pressedkeys

  # puts "generic::HandleDrop: $time"

  if {![info exists _drag_source] && ![string length $_drag_source]} {
    return refuse_drop
  }
  if {![info exists _drop_target] && ![string length $_drop_target]} {
    return refuse_drop
  }
  if {![llength $_common_drag_source_types]} {return refuse_drop}
  ## Get the dropped data.
  set data [GetDroppedData $time]
  ## Try to select the most specific <<Drop>> event.
  foreach type [concat $_common_drag_source_types $_common_drop_target_types] {
    set type [platform_independent_type $type]
    set cmd [bind $_drop_target <<Drop:$type>>]
    if {[string length $cmd]} {
      set cmd [string map [list %W $_drop_target %X $rootX %Y $rootY \
        %CST \{$_common_drag_source_types\} \
        %CTT \{$_common_drop_target_types\} \
        %CPT \{[lindex [platform_independent_type [lindex $_common_drag_source_types 0]] 0]\} \
        %ST  \{$_typelist\}    %TT \{$_types\} \
        %A   $_action          %a \{$_actionlist\} \
        %b   \{$_pressedkeys\} %m \{$_pressedkeys\} \
        %D   [list $data]      %e <<Drop:$type>> \
        %L   \{$_typelist\}    %% % \
        %t   \{$_typelist\}    %T  \{[lindex $_common_drag_source_types 0]\} \
        %c   \{$_codelist\}    %C  \{[lindex $_codelist 0]\} \
        ] $cmd]
      set _action [uplevel \#0 $cmd]
      # Return values: copy, move, link, ask, private, refuse_drop
      switch -exact -- $_action {
        copy - move - link - ask - private - refuse_drop - default {}
        default {set _action copy}
      }
      return $_action
    }
  }
  set cmd [bind $_drop_target <<Drop>>]
  if {[string length $cmd]} {
    set cmd [string map [list %W $_drop_target %X $rootX %Y $rootY \
      %CST \{$_common_drag_source_types\} \
      %CTT \{$_common_drop_target_types\} \
      %CPT \{[lindex [platform_independent_type [lindex $_common_drag_source_types 0]] 0]\} \
      %ST  \{$_typelist\}    %TT \{$_types\} \
      %A   $_action          %a \{$_actionlist\} \
      %b   \{$_pressedkeys\} %m \{$_pressedkeys\} \
      %D   [list $data]      %e <<Drop>> \
      %L   \{$_typelist\}    %% % \
      %t   \{$_typelist\}    %T  \{[lindex $_common_drag_source_types 0]\} \
      %c   \{$_codelist\}    %C  \{[lindex $_codelist 0]\} \
      ] $cmd]
    set _action [uplevel \#0 $cmd]
  }
  # Return values: copy, move, link, ask, private, refuse_drop
  switch -exact -- $_action {
    copy - move - link - ask - private - refuse_drop - default {}
    default {set _action copy}
  }
  return $_action
};# generic::HandleDrop

# ----------------------------------------------------------------------------
#  Command generic::GetWindowCommonTypes
# ----------------------------------------------------------------------------
proc generic::GetWindowCommonTypes { win typelist } {
  set types [bind $win <<DropTargetTypes>>]
  # debug ">> Accepted types: $win $_types"
  set common_drag_source_types {}
  set common_drop_target_types {}
  if {[llength $types]} {
    ## Examine the drop target types, to find at least one match with the drag
    ## source types...
    set supported_types [supported_types $typelist]
    foreach type $types {
      foreach matched [lsearch -glob -all -inline $supported_types $type] {
        ## Drop target supports this type.
        lappend common_drag_source_types $matched
        lappend common_drop_target_types $type
      }
    }
  }
  list $common_drag_source_types $common_drop_target_types
};# generic::GetWindowCommonTypes

# ----------------------------------------------------------------------------
#  Command generic::FindWindowWithCommonTypes
# ----------------------------------------------------------------------------
proc generic::FindWindowWithCommonTypes { win typelist } {
  set toplevel [winfo toplevel $win]
  while {![string equal $win $toplevel]} {
    foreach {common_drag_source_types common_drop_target_types} \
            [GetWindowCommonTypes $win $typelist] {break}
    if {[llength $common_drag_source_types]} {
      return [list $win $common_drag_source_types $common_drop_target_types]
    }
    set win [winfo parent $win]
  }
  ## We have reached the toplevel, which may be also a target (SF Bug #30)
  foreach {common_drag_source_types common_drop_target_types} \
          [GetWindowCommonTypes $win $typelist] {break}
  if {[llength $common_drag_source_types]} {
    return [list $win $common_drag_source_types $common_drop_target_types]
  }
  return { {} {} {} }
};# generic::FindWindowWithCommonTypes

# ----------------------------------------------------------------------------
#  Command generic::GetDroppedData
# ----------------------------------------------------------------------------
proc generic::GetDroppedData { time } {
  variable _dropped_data
  return  $_dropped_data
};# generic::GetDroppedData

# ----------------------------------------------------------------------------
#  Command generic::SetDroppedData
# ----------------------------------------------------------------------------
proc generic::SetDroppedData { data } {
  variable _dropped_data
  set _dropped_data $data
};# generic::SetDroppedData

# ----------------------------------------------------------------------------
#  Command generic::GetDragSource
# ----------------------------------------------------------------------------
proc generic::GetDragSource { } {
  variable _drag_source
  return  $_drag_source
};# generic::GetDragSource

# ----------------------------------------------------------------------------
#  Command generic::GetDropTarget
# ----------------------------------------------------------------------------
proc generic::GetDropTarget { } {
  variable _drop_target
  return $_drop_target
};# generic::GetDropTarget

# ----------------------------------------------------------------------------
#  Command generic::GetDragSourceCommonTypes
# ----------------------------------------------------------------------------
proc generic::GetDragSourceCommonTypes { } {
  variable _common_drag_source_types
  return  $_common_drag_source_types
};# generic::GetDragSourceCommonTypes

# ----------------------------------------------------------------------------
#  Command generic::GetDropTargetCommonTypes
# ----------------------------------------------------------------------------
proc generic::GetDropTargetCommonTypes { } {
  variable _common_drag_source_types
  return  $_common_drag_source_types
};# generic::GetDropTargetCommonTypes

# ----------------------------------------------------------------------------
#  Command generic::platform_specific_types
# ----------------------------------------------------------------------------
proc generic::platform_specific_types { types } {
  set new_types {}
  foreach type $types {
    set new_types [concat $new_types [platform_specific_type $type]]
  }
  return $new_types
}; # generic::platform_specific_types

# ----------------------------------------------------------------------------
#  Command generic::platform_specific_type
# ----------------------------------------------------------------------------
proc generic::platform_specific_type { type } {
  variable _tkdnd2platform
  if {[dict exists $_tkdnd2platform $type]} {
    return [dict get $_tkdnd2platform $type]
  }
  list $type
}; # generic::platform_specific_type

# ----------------------------------------------------------------------------
#  Command tkdnd::platform_independent_types
# ----------------------------------------------------------------------------
proc ::tkdnd::platform_independent_types { types } {
  set new_types {}
  foreach type $types {
    set new_types [concat $new_types [platform_independent_type $type]]
  }
  return $new_types
}; # tkdnd::platform_independent_types

# ----------------------------------------------------------------------------
#  Command generic::platform_independent_type
# ----------------------------------------------------------------------------
proc generic::platform_independent_type { type } {
  variable _platform2tkdnd
  if {[dict exists $_platform2tkdnd $type]} {
    return [dict get $_platform2tkdnd $type]
  }
  return $type
}; # generic::platform_independent_type

# ----------------------------------------------------------------------------
#  Command generic::supported_types
# ----------------------------------------------------------------------------
proc generic::supported_types { types } {
  set new_types {}
  foreach type $types {
    if {[supported_type $type]} {lappend new_types $type}
  }
  return $new_types
}; # generic::supported_types

# ----------------------------------------------------------------------------
#  Command generic::supported_type
# ----------------------------------------------------------------------------
proc generic::supported_type { type } {
  variable _platform2tkdnd
  if {[dict exists $_platform2tkdnd $type]} {
    return 1
  }
  return 0
}; # generic::supported_type
