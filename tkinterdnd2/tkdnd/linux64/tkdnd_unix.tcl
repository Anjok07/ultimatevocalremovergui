#
# tkdnd_unix.tcl --
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

namespace eval xdnd {
  variable _dragging 0

  proc initialise { } {
    ## Mapping from platform types to TkDND types...
    ::tkdnd::generic::initialise_platform_to_tkdnd_types [list \
       text/plain\;charset=utf-8     DND_Text  \
       UTF8_STRING                   DND_Text  \
       text/plain                    DND_Text  \
       STRING                        DND_Text  \
       TEXT                          DND_Text  \
       COMPOUND_TEXT                 DND_Text  \
       text/uri-list                 DND_Files \
       text/html\;charset=utf-8      DND_HTML  \
       text/html                     DND_HTML  \
       application/x-color           DND_Color \
    ]
  };# initialise

};# namespace xdnd

# ----------------------------------------------------------------------------
#  Command xdnd::HandleXdndEnter
# ----------------------------------------------------------------------------
proc xdnd::HandleXdndEnter { path drag_source typelist time { data {} } } {
  variable _pressedkeys
  variable _actionlist
  variable _typelist
  set _pressedkeys 1
  set _actionlist  { copy move link ask private }
  set _typelist    $typelist
  # puts "xdnd::HandleXdndEnter: $time"
  ::tkdnd::generic::SetDroppedData $data
  ::tkdnd::generic::HandleEnter $path $drag_source $typelist $typelist \
           $_actionlist $_pressedkeys
};# xdnd::HandleXdndEnter

# ----------------------------------------------------------------------------
#  Command xdnd::HandleXdndPosition
# ----------------------------------------------------------------------------
proc xdnd::HandleXdndPosition { drop_target rootX rootY time {drag_source {}} } {
  variable _pressedkeys
  variable _typelist
  variable _last_mouse_root_x; set _last_mouse_root_x $rootX
  variable _last_mouse_root_y; set _last_mouse_root_y $rootY
  # puts "xdnd::HandleXdndPosition: $time"
  ## Get the dropped data...
  catch {
    ::tkdnd::generic::SetDroppedData [GetPositionData $drop_target $_typelist $time]
  }
  ::tkdnd::generic::HandlePosition $drop_target $drag_source \
                                   $_pressedkeys $rootX $rootY
};# xdnd::HandleXdndPosition

# ----------------------------------------------------------------------------
#  Command xdnd::HandleXdndLeave
# ----------------------------------------------------------------------------
proc xdnd::HandleXdndLeave { } {
  ::tkdnd::generic::HandleLeave
};# xdnd::HandleXdndLeave

# ----------------------------------------------------------------------------
#  Command xdnd::_HandleXdndDrop
# ----------------------------------------------------------------------------
proc xdnd::HandleXdndDrop { time } {
  variable _pressedkeys
  variable _last_mouse_root_x
  variable _last_mouse_root_y
  ## Get the dropped data...
  ::tkdnd::generic::SetDroppedData [GetDroppedData \
    [::tkdnd::generic::GetDragSource] [::tkdnd::generic::GetDropTarget] \
    [::tkdnd::generic::GetDragSourceCommonTypes] $time]
  ::tkdnd::generic::HandleDrop {} {} $_pressedkeys \
                               $_last_mouse_root_x $_last_mouse_root_y $time
};# xdnd::HandleXdndDrop

# ----------------------------------------------------------------------------
#  Command xdnd::GetPositionData
# ----------------------------------------------------------------------------
proc xdnd::GetPositionData { drop_target typelist time } {
  foreach {drop_target common_drag_source_types common_drop_target_types} \
    [::tkdnd::generic::FindWindowWithCommonTypes $drop_target $typelist] {break}
  GetDroppedData [::tkdnd::generic::GetDragSource] $drop_target \
    $common_drag_source_types $time
};# xdnd::GetPositionData

# ----------------------------------------------------------------------------
#  Command xdnd::GetDroppedData
# ----------------------------------------------------------------------------
proc xdnd::GetDroppedData { _drag_source _drop_target _common_drag_source_types time } {
  if {![llength $_common_drag_source_types]} {
    error "no common data types between the drag source and drop target widgets"
  }
  ## Is drag source in this application?
  if {[catch {winfo pathname -displayof $_drop_target $_drag_source} p]} {
    set _use_tk_selection 0
  } else {
    set _use_tk_selection 1
  }
  foreach type $_common_drag_source_types {
    # puts "TYPE: $type ($_drop_target)"
    # _get_selection $_drop_target $time $type
    if {$_use_tk_selection} {
      if {![catch {
        selection get -displayof $_drop_target -selection XdndSelection \
                      -type $type
                                              } result options]} {
        return [normalise_data $type $result]
      }
    } else {
      # puts "_selection_get -displayof $_drop_target -selection XdndSelection \
      #                 -type $type -time $time"
      #after 100 [list focus -force $_drop_target]
      #after 50 [list raise [winfo toplevel $_drop_target]]
      if {![catch {
        _selection_get -displayof $_drop_target -selection XdndSelection \
                      -type $type -time $time
                                              } result options]} {
        return [normalise_data $type $result]
      }
    }
  }
  return -options $options $result
};# xdnd::GetDroppedData

# ----------------------------------------------------------------------------
#  Command xdnd::platform_specific_types
# ----------------------------------------------------------------------------
proc xdnd::platform_specific_types { types } {
  ::tkdnd::generic::platform_specific_types $types
}; # xdnd::platform_specific_types

# ----------------------------------------------------------------------------
#  Command xdnd::platform_specific_type
# ----------------------------------------------------------------------------
proc xdnd::platform_specific_type { type } {
  ::tkdnd::generic::platform_specific_type $type
}; # xdnd::platform_specific_type

# ----------------------------------------------------------------------------
#  Command tkdnd::platform_independent_types
# ----------------------------------------------------------------------------
proc ::tkdnd::platform_independent_types { types } {
  ::tkdnd::generic::platform_independent_types $types
}; # tkdnd::platform_independent_types

# ----------------------------------------------------------------------------
#  Command xdnd::platform_independent_type
# ----------------------------------------------------------------------------
proc xdnd::platform_independent_type { type } {
  ::tkdnd::generic::platform_independent_type $type
}; # xdnd::platform_independent_type

# ----------------------------------------------------------------------------
#  Command xdnd::_normalise_data
# ----------------------------------------------------------------------------
proc xdnd::normalise_data { type data } {
  # Tk knows how to interpret the following types:
  #    STRING, TEXT, COMPOUND_TEXT
  #    UTF8_STRING
  # Else, it returns a list of 8 or 32 bit numbers...
  switch -glob $type {
    STRING - UTF8_STRING - TEXT - COMPOUND_TEXT {return $data}
    text/html {
      if {[catch {
            encoding convertfrom unicode $data
           } string]} {
        set string $data
      }
      return [string map {\r\n \n} $string]
    }
    text/html\;charset=utf-8  -
    text/plain\;charset=utf-8 -
    text/plain {
      if {[catch {
            encoding convertfrom utf-8 [tkdnd::bytes_to_string $data]
           } string]} {
        set string $data
      }
      return [string map {\r\n \n} $string]
    }
    text/uri-list* {
      if {[catch {
            encoding convertfrom utf-8 [tkdnd::bytes_to_string $data]
          } string]} {
        set string $data
      }
      ## Get rid of \r\n
      set string [string trim [string map {\r\n \n} $string]]
      set files {}
      foreach quoted_file [split $string] {
        set file [tkdnd::urn_unquote $quoted_file]
        switch -glob $file {
          \#*       {}
          file://*  {lappend files [string range $file 7 end]}
          ftp://*   -
          https://* -
          http://*  {lappend files $quoted_file}
          default   {lappend files $file}
        }
      }
      return $files
    }
    application/x-color {
      return $data
    }
    text/x-moz-url -
    application/q-iconlist -
    default    {return $data}
  }
}; # xdnd::normalise_data

#############################################################################
##
##  XDND drag implementation
##
#############################################################################

# ----------------------------------------------------------------------------
#  Command xdnd::_selection_ownership_lost
# ----------------------------------------------------------------------------
proc xdnd::_selection_ownership_lost {} {
  variable _dragging
  set _dragging 0
};# _selection_ownership_lost

# ----------------------------------------------------------------------------
#  Command xdnd::_dodragdrop
# ----------------------------------------------------------------------------
proc xdnd::_dodragdrop { source actions types data button } {
  variable _dragging

  # puts "xdnd::_dodragdrop: source: $source, actions: $actions, types: $types,\
  #       data: \"$data\", button: $button"
  if {$_dragging} {
    ## We are in the middle of another drag operation...
    error "another drag operation in progress"
  }

  variable _dodragdrop_drag_source                $source
  variable _dodragdrop_drop_target                0
  variable _dodragdrop_drop_target_proxy          0
  variable _dodragdrop_actions                    $actions
  variable _dodragdrop_action_descriptions        $actions
  variable _dodragdrop_actions_len                [llength $actions]
  variable _dodragdrop_types                      $types
  variable _dodragdrop_types_len                  [llength $types]
  variable _dodragdrop_data                       $data
  variable _dodragdrop_transfer_data              {}
  variable _dodragdrop_button                     $button
  variable _dodragdrop_time                       0
  variable _dodragdrop_default_action             refuse_drop
  variable _dodragdrop_waiting_status             0
  variable _dodragdrop_drop_target_accepts_drop   0
  variable _dodragdrop_drop_target_accepts_action refuse_drop
  variable _dodragdrop_current_cursor             $_dodragdrop_default_action
  variable _dodragdrop_drop_occured               0
  variable _dodragdrop_selection_requestor        0

  ##
  ## If we have more than 3 types, the property XdndTypeList must be set on
  ## the drag source widget...
  ##
  if {$_dodragdrop_types_len > 3} {
    _announce_type_list $_dodragdrop_drag_source $_dodragdrop_types
  }

  ##
  ## Announce the actions & their descriptions on the XdndActionList &
  ## XdndActionDescription properties...
  ##
  _announce_action_list $_dodragdrop_drag_source $_dodragdrop_actions \
                        $_dodragdrop_action_descriptions

  ##
  ## Arrange selection handlers for our drag source, and all the supported types
  ##
  registerSelectionHandler $source $types

  ##
  ## Step 1: When a drag begins, the source takes ownership of XdndSelection.
  ##
  selection own -command ::tkdnd::xdnd::_selection_ownership_lost \
                -selection XdndSelection $source
  set _dragging 1

  ## Grab the mouse pointer...
  _grab_pointer $source $_dodragdrop_default_action

  ## Register our generic event handler...
  #  The generic event callback will report events by modifying variable
  #  ::xdnd::_dodragdrop_event: a dict with event information will be set as
  #  the value of the variable...
  _register_generic_event_handler

  ## Set a timeout for debugging purposes...
  #  after 60000 {set ::tkdnd::xdnd::_dragging 0}

  tkwait variable ::tkdnd::xdnd::_dragging
  _SendXdndLeave

  set _dragging 0
  _ungrab_pointer $source
  _unregister_generic_event_handler
  catch {selection clear -selection XdndSelection}
  unregisterSelectionHandler $source $types
  return $_dodragdrop_drop_target_accepts_action
};# xdnd::_dodragdrop

# ----------------------------------------------------------------------------
#  Command xdnd::_process_drag_events
# ----------------------------------------------------------------------------
proc xdnd::_process_drag_events {event} {
  # The return value from proc is normally 0. A non-zero return value indicates
  # that the event is not to be handled further; that is, proc has done all
  # processing that is to be allowed for the event
  variable _dragging
  if {!$_dragging} {return 0}
  # puts $event

  variable _dodragdrop_time
  set time [dict get $event time]
  set type [dict get $event type]
  if {$time < $_dodragdrop_time && ![string equal $type SelectionRequest]} {
    return 0
  }
  set _dodragdrop_time $time

  variable _dodragdrop_drag_source
  variable _dodragdrop_drop_target
  variable _dodragdrop_drop_target_proxy
  variable _dodragdrop_default_action
  switch $type {
    MotionNotify {
      set rootx  [dict get $event x_root]
      set rooty  [dict get $event y_root]
      set window [_find_drop_target_window $_dodragdrop_drag_source \
                                           $rootx $rooty]
      if {[string length $window]} {
        ## Examine the modifiers to suggest an action...
        set _dodragdrop_default_action [_default_action $event]
        ## Is it a Tk widget?
        # set path [winfo containing $rootx $rooty]
        # puts "Window under mouse: $window ($path)"
        if {$_dodragdrop_drop_target != $window} {
          ## Send XdndLeave to $_dodragdrop_drop_target
          _SendXdndLeave
          ## Is there a proxy? If not, _find_drop_target_proxy returns the
          ## target window, so we always get a valid "proxy".
          set proxy [_find_drop_target_proxy $_dodragdrop_drag_source $window]
          ## Send XdndEnter to $window
          _SendXdndEnter $window $proxy
          ## Send XdndPosition to $_dodragdrop_drop_target
          _SendXdndPosition $rootx $rooty $_dodragdrop_default_action
        } else {
          ## Send XdndPosition to $_dodragdrop_drop_target
          _SendXdndPosition $rootx $rooty $_dodragdrop_default_action
        }
      } else {
        ## No window under the mouse. Send XdndLeave to $_dodragdrop_drop_target
        _SendXdndLeave
      }
    }
    ButtonPress {
    }
    ButtonRelease {
      variable _dodragdrop_button
      set button [dict get $event button]
      if {$button == $_dodragdrop_button} {
        ## The button that initiated the drag was released. Trigger drop...
        _SendXdndDrop
      }
      return 1
    }
    KeyPress {
    }
    KeyRelease {
      set keysym [dict get $event keysym]
      switch $keysym {
        Escape {
          ## The user has pressed escape. Abort...
          if {$_dragging} {set _dragging 0}
        }
      }
    }
    SelectionRequest {
      variable _dodragdrop_selection_requestor
      variable _dodragdrop_selection_property
      variable _dodragdrop_selection_selection
      variable _dodragdrop_selection_target
      variable _dodragdrop_selection_time
      set _dodragdrop_selection_requestor [dict get $event requestor]
      set _dodragdrop_selection_property  [dict get $event property]
      set _dodragdrop_selection_selection [dict get $event selection]
      set _dodragdrop_selection_target    [dict get $event target]
      set _dodragdrop_selection_time      $time
      return 0
    }
    default {
      return 0
    }
  }
  return 0
};# _process_drag_events

# ----------------------------------------------------------------------------
#  Command xdnd::_SendXdndEnter
# ----------------------------------------------------------------------------
proc xdnd::_SendXdndEnter {window proxy} {
  variable _dodragdrop_drag_source
  variable _dodragdrop_drop_target
  variable _dodragdrop_drop_target_proxy
  variable _dodragdrop_types
  variable _dodragdrop_waiting_status
  variable _dodragdrop_drop_occured
  if {$_dodragdrop_drop_target > 0} _SendXdndLeave
  if {$_dodragdrop_drop_occured} return
  set _dodragdrop_drop_target       $window
  set _dodragdrop_drop_target_proxy $proxy
  set _dodragdrop_waiting_status    0
  if {$_dodragdrop_drop_target < 1} return
  # puts "XdndEnter: $_dodragdrop_drop_target $_dodragdrop_drop_target_proxy"
  _send_XdndEnter $_dodragdrop_drag_source $_dodragdrop_drop_target \
                  $_dodragdrop_drop_target_proxy $_dodragdrop_types
};# xdnd::_SendXdndEnter

# ----------------------------------------------------------------------------
#  Command xdnd::_SendXdndPosition
# ----------------------------------------------------------------------------
proc xdnd::_SendXdndPosition {rootx rooty action} {
  variable _dodragdrop_drag_source
  variable _dodragdrop_drop_target
  if {$_dodragdrop_drop_target < 1} return
  variable _dodragdrop_drop_occured
  if {$_dodragdrop_drop_occured} return
  variable _dodragdrop_drop_target_proxy
  variable _dodragdrop_waiting_status
  ## Arrange a new XdndPosition, to be send periodically...
  variable _dodragdrop_xdnd_position_heartbeat
  catch {after cancel $_dodragdrop_xdnd_position_heartbeat}
  set _dodragdrop_xdnd_position_heartbeat [after 200 \
    [list ::tkdnd::xdnd::_SendXdndPosition $rootx $rooty $action]]
  if {$_dodragdrop_waiting_status} {return}
  # puts "XdndPosition: $_dodragdrop_drop_target $rootx $rooty $action"
  _send_XdndPosition $_dodragdrop_drag_source $_dodragdrop_drop_target \
                     $_dodragdrop_drop_target_proxy $rootx $rooty $action
  set _dodragdrop_waiting_status 1
};# xdnd::_SendXdndPosition

# ----------------------------------------------------------------------------
#  Command xdnd::_HandleXdndStatus
# ----------------------------------------------------------------------------
proc xdnd::_HandleXdndStatus {event} {
  variable _dodragdrop_drop_target
  variable _dodragdrop_waiting_status

  variable _dodragdrop_drop_target_accepts_drop
  variable _dodragdrop_drop_target_accepts_action
  set _dodragdrop_waiting_status 0
  foreach key {target accept want_position action x y w h} {
    set $key [dict get $event $key]
  }
  set _dodragdrop_drop_target_accepts_drop   $accept
  set _dodragdrop_drop_target_accepts_action $action
  if {$_dodragdrop_drop_target < 1} return
  variable _dodragdrop_drop_occured
  if {$_dodragdrop_drop_occured} return
  _update_cursor
  # puts "XdndStatus: $event"
};# xdnd::_HandleXdndStatus

# ----------------------------------------------------------------------------
#  Command xdnd::_HandleXdndFinished
# ----------------------------------------------------------------------------
proc xdnd::_HandleXdndFinished {event} {
  variable _dodragdrop_xdnd_finished_event_after_id
  catch {after cancel $_dodragdrop_xdnd_finished_event_after_id}
  set _dodragdrop_xdnd_finished_event_after_id {}
  variable _dodragdrop_drop_target
  set _dodragdrop_drop_target 0
  variable _dragging
  if {$_dragging} {set _dragging 0}

  variable _dodragdrop_drop_target_accepts_drop
  variable _dodragdrop_drop_target_accepts_action
  if {[dict size $event]} {
    foreach key {target accept action} {
      set $key [dict get $event $key]
    }
    set _dodragdrop_drop_target_accepts_drop   $accept
    set _dodragdrop_drop_target_accepts_action $action
  } else {
    set _dodragdrop_drop_target_accepts_drop 0
  }
  if {!$_dodragdrop_drop_target_accepts_drop} {
    set _dodragdrop_drop_target_accepts_action refuse_drop
  }
  # puts "XdndFinished: $event"
};# xdnd::_HandleXdndFinished

# ----------------------------------------------------------------------------
#  Command xdnd::_SendXdndLeave
# ----------------------------------------------------------------------------
proc xdnd::_SendXdndLeave {} {
  variable _dodragdrop_drag_source
  variable _dodragdrop_drop_target
  if {$_dodragdrop_drop_target < 1} return
  variable _dodragdrop_drop_target_proxy
  # puts "XdndLeave: $_dodragdrop_drop_target"
  _send_XdndLeave $_dodragdrop_drag_source $_dodragdrop_drop_target \
                  $_dodragdrop_drop_target_proxy
  set _dodragdrop_drop_target 0
  variable _dodragdrop_drop_target_accepts_drop
  variable _dodragdrop_drop_target_accepts_action
  set _dodragdrop_drop_target_accepts_drop   0
  set _dodragdrop_drop_target_accepts_action refuse_drop
  variable _dodragdrop_drop_occured
  if {$_dodragdrop_drop_occured} return
  _update_cursor
};# xdnd::_SendXdndLeave

# ----------------------------------------------------------------------------
#  Command xdnd::_SendXdndDrop
# ----------------------------------------------------------------------------
proc xdnd::_SendXdndDrop {} {
  variable _dodragdrop_drag_source
  variable _dodragdrop_drop_target
  if {$_dodragdrop_drop_target < 1} {
    ## The mouse has been released over a widget that does not accept drops.
    _HandleXdndFinished {}
    return
  }
  variable _dodragdrop_drop_occured
  if {$_dodragdrop_drop_occured} {return}
  variable _dodragdrop_drop_target_proxy
  variable _dodragdrop_drop_target_accepts_drop
  variable _dodragdrop_drop_target_accepts_action

  set _dodragdrop_drop_occured 1
  _update_cursor clock

  if {!$_dodragdrop_drop_target_accepts_drop} {
    _SendXdndLeave
    _HandleXdndFinished {}
    return
  }
  # puts "XdndDrop: $_dodragdrop_drop_target"
  variable _dodragdrop_drop_timestamp
  set _dodragdrop_drop_timestamp [_send_XdndDrop \
                 $_dodragdrop_drag_source $_dodragdrop_drop_target \
                 $_dodragdrop_drop_target_proxy]
  set _dodragdrop_drop_target 0
  # puts "XdndDrop: $_dodragdrop_drop_target"
  ## Arrange a timeout for receiving XdndFinished...
  variable _dodragdrop_xdnd_finished_event_after_id
  set _dodragdrop_xdnd_finished_event_after_id \
    [after 10000 [list ::tkdnd::xdnd::_HandleXdndFinished {}]]
};# xdnd::_SendXdndDrop

# ----------------------------------------------------------------------------
#  Command xdnd::_update_cursor
# ----------------------------------------------------------------------------
proc xdnd::_update_cursor { {cursor {}}} {
  # puts "_update_cursor $cursor"
  variable _dodragdrop_current_cursor
  variable _dodragdrop_drag_source
  variable _dodragdrop_drop_target_accepts_drop
  variable _dodragdrop_drop_target_accepts_action

  if {![string length $cursor]} {
    set cursor refuse_drop
    if {$_dodragdrop_drop_target_accepts_drop} {
      set cursor $_dodragdrop_drop_target_accepts_action
    }
  }
  if {![string equal $cursor $_dodragdrop_current_cursor]} {
    _set_pointer_cursor $_dodragdrop_drag_source $cursor
    set _dodragdrop_current_cursor $cursor
  }
};# xdnd::_update_cursor

# ----------------------------------------------------------------------------
#  Command xdnd::_default_action
# ----------------------------------------------------------------------------
proc xdnd::_default_action {event} {
  variable _dodragdrop_actions
  variable _dodragdrop_actions_len
  if {$_dodragdrop_actions_len == 1} {return [lindex $_dodragdrop_actions 0]}

  set alt     [dict get $event Alt]
  set shift   [dict get $event Shift]
  set control [dict get $event Control]

  if {$shift && $control && [lsearch $_dodragdrop_actions link] != -1} {
    return link
  } elseif {$control && [lsearch $_dodragdrop_actions copy] != -1} {
    return copy
  } elseif {$shift && [lsearch $_dodragdrop_actions move] != -1} {
    return move
  } elseif {$alt && [lsearch $_dodragdrop_actions link] != -1} {
    return link
  }
  return default
};# xdnd::_default_action

# ----------------------------------------------------------------------------
#  Command xdnd::getFormatForType
# ----------------------------------------------------------------------------
proc xdnd::getFormatForType {type} {
  switch -glob [string tolower $type] {
    text/plain\;charset=utf-8 -
    text/html\;charset=utf-8  -
    utf8_string               {set format UTF8_STRING}
    text/html                 -
    text/plain                -
    string                    -
    text                      -
    compound_text             {set format STRING}
    text/uri-list*            {set format UTF8_STRING}
    application/x-color       {set format $type}
    default                   {set format $type}
  }
  return $format
};# xdnd::getFormatForType

# ----------------------------------------------------------------------------
#  Command xdnd::registerSelectionHandler
# ----------------------------------------------------------------------------
proc xdnd::registerSelectionHandler {source types} {
  foreach type $types {
    selection handle -selection XdndSelection \
                     -type $type \
                     -format [getFormatForType $type] \
                     $source [list ::tkdnd::xdnd::_SendData $type]
  }
};# xdnd::registerSelectionHandler

# ----------------------------------------------------------------------------
#  Command xdnd::unregisterSelectionHandler
# ----------------------------------------------------------------------------
proc xdnd::unregisterSelectionHandler {source types} {
  foreach type $types {
    catch {
      selection handle -selection XdndSelection \
                       -type $type \
                       -format [getFormatForType $type] \
                       $source {}
    }
  }
};# xdnd::unregisterSelectionHandler

# ----------------------------------------------------------------------------
#  Command xdnd::_convert_to_unsigned
# ----------------------------------------------------------------------------
proc xdnd::_convert_to_unsigned {data format} {
  switch $format {
    8  { set mask 0xff }
    16 { set mask 0xffff }
    32 { set mask 0xffffff }
    default {error "unsupported format $format"}
  }
  ## Convert signed integer into unsigned...
  set d [list]
  foreach num $data {
    lappend d [expr { $num & $mask }]
  }
  return $d
};# xdnd::_convert_to_unsigned

# ----------------------------------------------------------------------------
#  Command xdnd::_SendData
# ----------------------------------------------------------------------------
proc xdnd::_SendData {type offset bytes args} {
  variable _dodragdrop_drag_source
  variable _dodragdrop_types
  variable _dodragdrop_data
  variable _dodragdrop_transfer_data

  ## The variable _dodragdrop_data contains a list of data, one for each
  ## type in the _dodragdrop_types variable. We have to search types, and find
  ## the corresponding entry in the _dodragdrop_data list.
  set index [lsearch $_dodragdrop_types $type]
  if {$index < 0} {
    error "unable to locate data suitable for type \"$type\""
  }
  set typed_data [lindex $_dodragdrop_data $index]
  set format 8
  if {$offset == 0} {
    ## Prepare the data to be transferred...
    switch -glob $type {
      text/plain* - UTF8_STRING - STRING - TEXT - COMPOUND_TEXT {
        binary scan [encoding convertto utf-8 $typed_data] \
                    c* _dodragdrop_transfer_data
        set _dodragdrop_transfer_data \
           [_convert_to_unsigned $_dodragdrop_transfer_data $format]
      }
      text/uri-list* {
        set files [list]
        foreach file $typed_data {
          switch -glob $file {
            *://*     {lappend files $file}
            default   {lappend files file://$file}
          }
        }
        binary scan [encoding convertto utf-8 "[join $files \r\n]\r\n"] \
                    c* _dodragdrop_transfer_data
        set _dodragdrop_transfer_data \
           [_convert_to_unsigned $_dodragdrop_transfer_data $format]
      }
      application/x-color {
        set format 16
        ## Try to understand the provided data: we accept a standard Tk colour,
        ## or a list of 3 values (red green blue) or a list of 4 values
        ## (red green blue opacity).
        switch [llength $typed_data] {
          1 { set color [winfo rgb $_dodragdrop_drag_source $typed_data]
              lappend color 65535 }
          3 { set color $typed_data; lappend color 65535 }
          4 { set color $typed_data }
          default {error "unknown color data: \"$typed_data\""}
        }
        ## Convert the 4 elements into 16 bit values...
        set _dodragdrop_transfer_data [list]
        foreach c $color {
          lappend _dodragdrop_transfer_data [format 0x%04X $c]
        }
      }
      default {
        set format 32
        binary scan $typed_data c* _dodragdrop_transfer_data
      }
    }
  }

  ##
  ## Data has been split into bytes. Count the bytes requested, and return them
  ##
  set data [lrange $_dodragdrop_transfer_data $offset [expr {$offset+$bytes-1}]]
  switch $format {
    8  {
      set data [encoding convertfrom utf-8 [binary format c* $data]]
    }
    16 {
      variable _dodragdrop_selection_requestor
      if {$_dodragdrop_selection_requestor} {
        ## Tk selection cannot process this format (only 8 & 32 supported).
        ## Call our XChangeProperty...
        set numItems [llength $data]
        variable _dodragdrop_selection_property
        variable _dodragdrop_selection_selection
        variable _dodragdrop_selection_target
        variable _dodragdrop_selection_time
        XChangeProperty $_dodragdrop_drag_source \
                        $_dodragdrop_selection_requestor \
                        $_dodragdrop_selection_property \
                        $_dodragdrop_selection_target \
                        $format \
                        $_dodragdrop_selection_time \
                        $data $numItems
        return -code break
      }
    }
    32 {
    }
    default {
      error "unsupported format $format"
    }
  }
  # puts "SendData: $type $offset $bytes $args ($typed_data)"
  # puts "          $data"
  return $data
};# xdnd::_SendData
