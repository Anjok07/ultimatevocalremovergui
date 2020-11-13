#
# tkdnd_windows.tcl --
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

namespace eval olednd {

  proc initialise { } {
    ## Mapping from platform types to TkDND types...
    ::tkdnd::generic::initialise_platform_to_tkdnd_types [list \
       CF_UNICODETEXT          DND_Text  \
       CF_TEXT                 DND_Text  \
       CF_HDROP                DND_Files \
       UniformResourceLocator  DND_URL   \
       CF_HTML                 DND_HTML  \
       {HTML Format}           DND_HTML  \
       CF_RTF                  DND_RTF   \
       CF_RTFTEXT              DND_RTF   \
       {Rich Text Format}      DND_RTF   \
    ]
    # FileGroupDescriptorW    DND_Files \
    # FileGroupDescriptor     DND_Files \

    ## Mapping from TkDND types to platform types...
    ::tkdnd::generic::initialise_tkdnd_to_platform_types [list \
       DND_Text  {CF_UNICODETEXT CF_TEXT}               \
       DND_Files {CF_HDROP}                             \
       DND_URL   {UniformResourceLocator UniformResourceLocatorW} \
       DND_HTML  {CF_HTML {HTML Format}}                \
       DND_RTF   {CF_RTF CF_RTFTEXT {Rich Text Format}} \
    ]
  };# initialise

};# namespace olednd

# ----------------------------------------------------------------------------
#  Command olednd::HandleDragEnter
# ----------------------------------------------------------------------------
proc olednd::HandleDragEnter { drop_target typelist actionlist pressedkeys
                               rootX rootY codelist { data {} } } {
  ::tkdnd::generic::SetDroppedData $data
  focus $drop_target
  ::tkdnd::generic::HandleEnter $drop_target 0 $typelist \
                                $codelist $actionlist $pressedkeys
  set action [::tkdnd::generic::HandlePosition $drop_target {} \
                                               $pressedkeys $rootX $rootY]
  if {$::tkdnd::_auto_update} {update idletasks}
  return $action
};# olednd::HandleDragEnter

# ----------------------------------------------------------------------------
#  Command olednd::HandleDragOver
# ----------------------------------------------------------------------------
proc olednd::HandleDragOver { drop_target pressedkeys rootX rootY } {
  set action [::tkdnd::generic::HandlePosition $drop_target {} \
                                               $pressedkeys $rootX $rootY]
  if {$::tkdnd::_auto_update} {update idletasks}
  return $action
};# olednd::HandleDragOver

# ----------------------------------------------------------------------------
#  Command olednd::HandleDragLeave
# ----------------------------------------------------------------------------
proc olednd::HandleDragLeave { drop_target } {
  ::tkdnd::generic::HandleLeave
  if {$::tkdnd::_auto_update} {update idletasks}
};# olednd::HandleDragLeave

# ----------------------------------------------------------------------------
#  Command olednd::HandleDrop
# ----------------------------------------------------------------------------
proc olednd::HandleDrop { drop_target pressedkeys rootX rootY type data } {
  ::tkdnd::generic::SetDroppedData [normalise_data $type $data]
  set action [::tkdnd::generic::HandleDrop $drop_target {} \
                                           $pressedkeys $rootX $rootY 0]
  if {$::tkdnd::_auto_update} {update idletasks}
  return $action
};# olednd::HandleDrop

# ----------------------------------------------------------------------------
#  Command olednd::GetDataType
# ----------------------------------------------------------------------------
proc olednd::GetDataType { drop_target typelist } {
  foreach {drop_target common_drag_source_types common_drop_target_types} \
    [::tkdnd::generic::FindWindowWithCommonTypes $drop_target $typelist] {break}
  lindex $common_drag_source_types 0
};# olednd::GetDataType

# ----------------------------------------------------------------------------
#  Command olednd::GetDragSourceCommonTypes
# ----------------------------------------------------------------------------
proc olednd::GetDragSourceCommonTypes { drop_target } {
  ::tkdnd::generic::GetDragSourceCommonTypes
};# olednd::GetDragSourceCommonTypes

# ----------------------------------------------------------------------------
#  Command olednd::platform_specific_types
# ----------------------------------------------------------------------------
proc olednd::platform_specific_types { types } {
  ::tkdnd::generic::platform_specific_types $types
}; # olednd::platform_specific_types

# ----------------------------------------------------------------------------
#  Command olednd::platform_specific_type
# ----------------------------------------------------------------------------
proc olednd::platform_specific_type { type } {
  ::tkdnd::generic::platform_specific_type $type
}; # olednd::platform_specific_type

# ----------------------------------------------------------------------------
#  Command tkdnd::platform_independent_types
# ----------------------------------------------------------------------------
proc ::tkdnd::platform_independent_types { types } {
  ::tkdnd::generic::platform_independent_types $types
}; # tkdnd::platform_independent_types

# ----------------------------------------------------------------------------
#  Command olednd::platform_independent_type
# ----------------------------------------------------------------------------
proc olednd::platform_independent_type { type } {
  ::tkdnd::generic::platform_independent_type $type
}; # olednd::platform_independent_type

# ----------------------------------------------------------------------------
#  Command olednd::normalise_data
# ----------------------------------------------------------------------------
proc olednd::normalise_data { type data } {
  switch [lindex [::tkdnd::generic::platform_independent_type $type] 0] {
    DND_Text   {return $data}
    DND_Files  {return $data}
    DND_HTML   {return [encoding convertfrom utf-8 $data]}
    default    {return $data}
  }
}; # olednd::normalise_data
