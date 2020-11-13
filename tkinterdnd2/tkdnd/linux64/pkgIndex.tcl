#
# Tcl package index file
#
package ifneeded tkdnd 2.9.2 \
  "source \{$dir/tkdnd.tcl\} ; \
   tkdnd::initialise \{$dir\} libtkdnd2.9.2.so tkdnd"

package ifneeded tkdnd::utils 2.9.2 \
  "source \{$dir/tkdnd_utils.tcl\} ; \
   package provide tkdnd::utils 2.9.2"
