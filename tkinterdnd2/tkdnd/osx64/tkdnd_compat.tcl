#
# tkdnd_compat.tcl --
# 
#    This file implements some utility procedures, to support older versions
#    of the TkDND package.
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

namespace eval compat {

};# namespace compat

# ----------------------------------------------------------------------------
#  Command ::dnd
# ----------------------------------------------------------------------------
proc ::dnd {method window args} {
  switch $method {
    bindtarget {
      switch [llength $args] {
        0 {return [tkdnd::compat::bindtarget0 $window]}
        1 {return [tkdnd::compat::bindtarget1 $window [lindex $args 0]]}
        2 {return [tkdnd::compat::bindtarget2 $window [lindex $args 0] \
                                                      [lindex $args 1]]}
        3 {return [tkdnd::compat::bindtarget3 $window [lindex $args 0] \
                                     [lindex $args 1] [lindex $args 2]]}
        4 {return [tkdnd::compat::bindtarget4 $window [lindex $args 0] \
                    [lindex $args 1] [lindex $args 2] [lindex $args 3]]}
      }
    }
    cleartarget {
      return [tkdnd::compat::cleartarget $window]
    }
    bindsource {
      switch [llength $args] {
        0 {return [tkdnd::compat::bindsource0 $window]}
        1 {return [tkdnd::compat::bindsource1 $window [lindex $args 0]]}
        2 {return [tkdnd::compat::bindsource2 $window [lindex $args 0] \
                                                      [lindex $args 1]]}
        3 {return [tkdnd::compat::bindsource3 $window [lindex $args 0] \
                                     [lindex $args 1] [lindex $args 2]]}
      }
    }
    clearsource {
      return [tkdnd::compat::clearsource $window]
    }
    drag {
      return [tkdnd::_init_drag 1 $window "press" 0 0 0 0]
    }
  }
  error "invalid number of arguments!"
};# ::dnd

# ----------------------------------------------------------------------------
#  Command compat::bindtarget
# ----------------------------------------------------------------------------
proc compat::bindtarget0 {window} {
  return [bind $window <<DropTargetTypes>>]
};# compat::bindtarget0

proc compat::bindtarget1 {window type} {
  return [bindtarget2 $window $type <Drop>]
};# compat::bindtarget1

proc compat::bindtarget2 {window type event} {
  switch $event {
    <DragEnter> {return [bind $window <<DropEnter>>]}
    <Drag>      {return [bind $window <<DropPosition>>]}
    <DragLeave> {return [bind $window <<DropLeave>>]}
    <Drop>      {return [bind $window <<Drop>>]}
  }
};# compat::bindtarget2

proc compat::bindtarget3 {window type event script} {
  set type [normalise_type $type]
  ::tkdnd::drop_target register $window [list $type]
  switch $event {
    <DragEnter> {return [bind $window <<DropEnter>> $script]}
    <Drag>      {return [bind $window <<DropPosition>> $script]}
    <DragLeave> {return [bind $window <<DropLeave>> $script]}
    <Drop>      {return [bind $window <<Drop>> $script]}
  }
};# compat::bindtarget3

proc compat::bindtarget4 {window type event script priority} {
  return [bindtarget3 $window $type $event $script]
};# compat::bindtarget4

proc compat::normalise_type { type } {
  switch $type {
    text/plain -
    {text/plain;charset=UTF-8} -
    Text                       {return DND_Text}
    text/uri-list -
    Files                      {return DND_Files}
    default                    {return $type}
  }
};# compat::normalise_type

# ----------------------------------------------------------------------------
#  Command compat::bindsource
# ----------------------------------------------------------------------------
proc compat::bindsource0 {window} {
  return [bind $window <<DropTargetTypes>>]
};# compat::bindsource0

proc compat::bindsource1 {window type} {
  return [bindsource2 $window $type <Drop>]
};# compat::bindsource1

proc compat::bindsource2 {window type script} {
  set type [normalise_type $type]
  ::tkdnd::drag_source register $window $type
  bind $window <<DragInitCmd>> "list {copy} {%t} \[$script\]"
};# compat::bindsource2

proc compat::bindsource3 {window type script priority} {
  return [bindsource2 $window $type $script]
};# compat::bindsource3

# ----------------------------------------------------------------------------
#  Command compat::cleartarget
# ----------------------------------------------------------------------------
proc compat::cleartarget {window} {
};# compat::cleartarget

# ----------------------------------------------------------------------------
#  Command compat::clearsource
# ----------------------------------------------------------------------------
proc compat::clearsource {window} {
};# compat::clearsource
