#
# tkdnd_macosx.tcl --
#
#    This file implements some utility procedures that are used by the TkDND
#    package.

#   This software is copyrighted by:
#   Georgios Petasis, Athens, Greece.
#   e-mail: petasisg@yahoo.gr, petasis@iit.demokritos.gr
#
#   Mac portions (c) 2009 Kevin Walzer/WordTech Communications LLC,
#   kw@codebykevin.com
#
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

#basic API for Mac Drag and Drop

#two data types supported: strings and file paths

#two commands at C level: ::tkdnd::macdnd::registerdragwidget, ::tkdnd::macdnd::unregisterdragwidget

#data retrieval mechanism: text or file paths are copied from drag clipboard to system clipboard and retrieved via [clipboard get]; array of file paths is converted to single tab-separated string, can be split into Tcl list

if {[tk windowingsystem] eq "aqua" && "AppKit" ni [winfo server .]} {
  error {TkAqua Cocoa required}
}

namespace eval macdnd {

  proc initialise { } {
     ## Mapping from platform types to TkDND types...
    ::tkdnd::generic::initialise_platform_to_tkdnd_types [list \
       NSPasteboardTypeString  DND_Text  \
       NSFilenamesPboardType   DND_Files \
       NSPasteboardTypeHTML    DND_HTML  \
    ]
  };# initialise

};# namespace macdnd

# ----------------------------------------------------------------------------
#  Command macdnd::HandleEnter
# ----------------------------------------------------------------------------
proc macdnd::HandleEnter { path drag_source typelist { data {} } } {
  variable _pressedkeys
  variable _actionlist
  set _pressedkeys 1
  set _actionlist  { copy move link ask private }
  ::tkdnd::generic::SetDroppedData $data
  ::tkdnd::generic::HandleEnter $path $drag_source $typelist $typelist \
           $_actionlist $_pressedkeys
};# macdnd::HandleEnter

# ----------------------------------------------------------------------------
#  Command macdnd::HandlePosition
# ----------------------------------------------------------------------------
proc macdnd::HandlePosition { drop_target rootX rootY {drag_source {}} } {
  variable _pressedkeys
  variable _last_mouse_root_x; set _last_mouse_root_x $rootX
  variable _last_mouse_root_y; set _last_mouse_root_y $rootY
  ::tkdnd::generic::HandlePosition $drop_target $drag_source \
                                   $_pressedkeys $rootX $rootY
};# macdnd::HandlePosition

# ----------------------------------------------------------------------------
#  Command macdnd::HandleLeave
# ----------------------------------------------------------------------------
proc macdnd::HandleLeave { args } {
  ::tkdnd::generic::HandleLeave
};# macdnd::HandleLeave

# ----------------------------------------------------------------------------
#  Command macdnd::HandleDrop
# ----------------------------------------------------------------------------
proc macdnd::HandleDrop { drop_target data args } {
  variable _pressedkeys
  variable _last_mouse_root_x
  variable _last_mouse_root_y
  ## Get the dropped data...
  ::tkdnd::generic::SetDroppedData $data
  ::tkdnd::generic::HandleDrop {} {} $_pressedkeys \
                               $_last_mouse_root_x $_last_mouse_root_y 0
};# macdnd::HandleDrop

# ----------------------------------------------------------------------------
#  Command macdnd::GetDragSourceCommonTypes
# ----------------------------------------------------------------------------
proc macdnd::GetDragSourceCommonTypes { } {
  ::tkdnd::generic::GetDragSourceCommonTypes
};# macdnd::GetDragSourceCommonTypes

# ----------------------------------------------------------------------------
#  Command macdnd::platform_specific_types
# ----------------------------------------------------------------------------
proc macdnd::platform_specific_types { types } {
  ::tkdnd::generic::platform_specific_types $types
}; # macdnd::platform_specific_types

# ----------------------------------------------------------------------------
#  Command macdnd::platform_specific_type
# ----------------------------------------------------------------------------
proc macdnd::platform_specific_type { type } {
  ::tkdnd::generic::platform_specific_type $type
}; # macdnd::platform_specific_type

# ----------------------------------------------------------------------------
#  Command tkdnd::platform_independent_types
# ----------------------------------------------------------------------------
proc ::tkdnd::platform_independent_types { types } {
  ::tkdnd::generic::platform_independent_types $types
}; # tkdnd::platform_independent_types

# ----------------------------------------------------------------------------
#  Command macdnd::platform_independent_type
# ----------------------------------------------------------------------------
proc macdnd::platform_independent_type { type } {
  ::tkdnd::generic::platform_independent_type $type
}; # macdnd::platform_independent_type
