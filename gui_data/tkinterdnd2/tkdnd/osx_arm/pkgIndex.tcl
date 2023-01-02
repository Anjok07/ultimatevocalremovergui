#
# Tcl package index file
#

namespace eval ::tkdnd {
  ## Check if a debug level must be set...
  if {[info exists ::TKDND_DEBUG_LEVEL]} {
    variable _debug_level $::TKDND_DEBUG_LEVEL
  } elseif {[info exists ::env(TKDND_DEBUG_LEVEL)]} {
    variable _debug_level $::env(TKDND_DEBUG_LEVEL)
  } else {
    variable _debug_level 0
  }

  # ----------------------------------------------------------------------------
  #  Command tkdnd::debug_enabled: returns the requested debug level (0 = no debug).
  # ----------------------------------------------------------------------------
  proc debug_enabled { {level {}} } {
    variable _debug_level
    if {$level != {}} {
      if {[string is integer -strict $level]} {
        set _debug_level $level
      } elseif {[string is true $level]} {
        set _debug_level 1
      }
    }
    return $_debug_level
  };# debug_enabled

  # ----------------------------------------------------------------------------
  #  Command tkdnd::source: source a Tcl fileInitialise the TkDND package.
  # ----------------------------------------------------------------------------
  proc source { filename { encoding utf-8 } } {
    variable _package_dir
    # If in debug mode, enable debug statements...
    set dbg_lvl [debug_enabled]
    if {$dbg_lvl} {
      puts "tkdnd::source (debug level $dbg_lvl) $filename"
      set fd [open $filename r]
      fconfigure $fd -encoding $encoding
      set script [read $fd]
      close $fd
      set map {}
      for {set lvl 0} {$lvl <= $dbg_lvl} {incr lvl} {
        lappend map "#DBG$lvl " {}
      }
      lappend map {#DBG } {}
      set script [string map $map $script]
      return [eval $script]
    }
    ::source -encoding $encoding $filename
  };# source

}; # namespace ::tkdnd

package ifneeded tkdnd 2.9.3 \
  "tkdnd::source \{$dir/tkdnd.tcl\} ; \
   tkdnd::initialise \{$dir\} libtkdnd2.9.3.dylib tkdnd"

package ifneeded tkdnd::utils 2.9.3 \
  "tkdnd::source \{$dir/tkdnd_utils.tcl\} ; \
   package provide tkdnd::utils 2.9.3"
