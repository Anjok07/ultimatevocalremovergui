# Copyright © 2021 rdbende <rdbende@gmail.com>

source [file join [file dirname [info script]] theme dark.tcl]

option add *tearOff 0

proc set_theme {mode} {
	if {$mode == "dark"} {
		ttk::style theme use "sun-valley-dark"

        set fontString "$::fontName"
        set fgSet "$::fgcolorset"

		array set colors {
		    -fg             "#F6F6F7"
		    -bg             "#0e0e0f"
		    -disabledfg     "#F6F6F7"
		    -selectfg       "#F6F6F7"
		    -selectbg       "#003b50"
		}
        
        ttk::style configure . \
            -background $colors(-bg) \
            -foreground $fgSet \
            -troughcolor $colors(-bg) \
            -focuscolor $colors(-selectbg) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -insertwidth 0 \
            -insertcolor $colors(-fg) \
            -fieldbackground $colors(-selectbg) \
            -font $fontString \
            -borderwidth 0 \
            -relief flat

        tk_setPalette \
        	background [ttk::style lookup . -background] \
            foreground [ttk::style lookup . -foreground] \
            highlightColor [ttk::style lookup . -focuscolor] \
            selectBackground [ttk::style lookup . -selectbackground] \
            selectForeground [ttk::style lookup . -selectforeground] \
            activeBackground [ttk::style lookup . -selectbackground] \
            activeForeground [ttk::style lookup . -selectforeground]
        
        ttk::style map . -foreground [list disabled $colors(-disabledfg)]

        option add *font [ttk::style lookup . -font]
        option add *Menu.selectcolor $colors(-fg)
        option add *Menu.background #0e0e0f
    
	} 
}
