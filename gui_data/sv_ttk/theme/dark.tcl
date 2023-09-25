# Copyright © 2021 rdbende <rdbende@gmail.com>

# A stunning dark theme for ttk based on Microsoft's Sun Valley visual style 

package require Tk 8.6

namespace eval ttk::theme::sun-valley-dark {
    variable version 1.0
    package provide ttk::theme::sun-valley-dark $version

    ttk::style theme create sun-valley-dark -parent clam -settings {
        proc load_images {imgdir} {
            variable images
            foreach file [glob -directory $imgdir *.png] {
                set images([file tail [file rootname $file]]) \
                [image create photo -file $file -format png]
            }
        }

        load_images [file join [file dirname [info script]] dark]

        array set colors {
            -fg             "#F6F6F7"
            -bg             "#0e0e0f"
            -disabledfg     "#F6F6F7"
            -selectfg       "#ffffff"
            -selectbg       "#2f60d8"
        }

        ttk::style layout TButton {
            Button.button -children {
                Button.padding -children {
                    Button.label -side left -expand 1
                } 
            }
        }

        ttk::style layout Toolbutton {
            Toolbutton.button -children {
                Toolbutton.padding -children {
                    Toolbutton.label -side left -expand 1
                } 
            }
        }

        ttk::style layout TMenubutton {
            Menubutton.button -children {
                Menubutton.padding -children {
                    Menubutton.label -side left -expand 1
                    Menubutton.indicator -side right -sticky nsew
                }
            }
        }

        ttk::style layout TOptionMenu {
            OptionMenu.button -children {
                OptionMenu.padding -children {
                    OptionMenu.label -side left -expand 0
                    OptionMenu.indicator -side right -sticky nsew
                }
            }
        }

        ttk::style layout Accent.TButton {
            AccentButton.button -children {
                AccentButton.padding -children {
                    AccentButton.label -side left -expand 1
                } 
            }
        }

        ttk::style layout Titlebar.TButton {
            TitlebarButton.button -children {
                TitlebarButton.padding -children {
                    TitlebarButton.label -side left -expand 1
                } 
            }
        }

        ttk::style layout Close.Titlebar.TButton {
            CloseButton.button -children {
                CloseButton.padding -children {
                    CloseButton.label -side left -expand 1
                } 
            }
        }

        ttk::style layout TCheckbutton {
            Checkbutton.button -children {
                Checkbutton.padding -children {
                    Checkbutton.indicator -side left
                    Checkbutton.label -side right -expand 1
                }
            }
        }

        ttk::style layout Switch.TCheckbutton {
            Switch.button -children {
                Switch.padding -children {
                    Switch.indicator -side left
                    Switch.label -side right -expand 1
                }
            }
        }

        ttk::style layout Toggle.TButton {
            ToggleButton.button -children {
                ToggleButton.padding -children {
                    ToggleButton.label -side left -expand 1
                } 
            }
        }

        ttk::style layout TRadiobutton {
            Radiobutton.button -children {
                Radiobutton.padding -children {
                    Radiobutton.indicator -side left
                    Radiobutton.label -side right -expand 1
                }
            }
        }

        ttk::style layout Vertical.TScrollbar {
            Vertical.Scrollbar.trough -sticky ns -children {
                Vertical.Scrollbar.uparrow -side top
                Vertical.Scrollbar.downarrow -side bottom
                Vertical.Scrollbar.thumb -expand 1
            }
        }

        ttk::style layout Horizontal.TScrollbar {
            Horizontal.Scrollbar.trough -sticky ew -children {
                Horizontal.Scrollbar.leftarrow -side left
                Horizontal.Scrollbar.rightarrow -side right
                Horizontal.Scrollbar.thumb -expand 1
            }
        }

        ttk::style layout TSeparator {
            TSeparator.separator -sticky nsew
        }


        # # Modify the TCombobox style
        # ttk::style configure TCombobox -borderwidth 3

        # # Define the layout of the ThickBorder.TCombobox
        # ttk::style layout ThickBorder.TCombobox {
        #     Combobox.field -sticky nsew -children {
        #         Combobox.padding -expand 1 -sticky nsew -children {
        #             Combobox.textarea -sticky nsew
        #         }
        #     }
        #     null -side right -sticky ns -children {
        #         Combobox.arrow -sticky nsew
        #     }
        # }

        # # Use a canvas as the parent of the combobox and create a custom border
        # canvas .c -width 200 -height 30 -highlightthickness 0
        # canvas .c create rectangle 2 2 198 28 -width 3 -outline black
        # pack .c
        # ttk::combobox .c.cbox -values {"Option 1" "Option 2" "Option 3"} -style ThickBorder.TCombobox
        # .c create window 100 15 -window .c.cbox

        ttk::style layout TCombobox {
            Combobox.field -sticky nsew -children {
                Combobox.padding -expand 1 -sticky nsew -children {
                    Combobox.textarea -sticky nsew
                }
            }
            null -side right -sticky ns -children {
                Combobox.arrow -sticky nsew
            }
        }
        
        ttk::style layout TSpinbox {
            Spinbox.field -sticky nsew -children {
                Spinbox.padding -expand 1 -sticky nsew -children {
                    Spinbox.textarea -sticky nsew
                }
                
            }
            null -side right -sticky nsew -children {
                Spinbox.uparrow -side left -sticky nsew
                Spinbox.downarrow -side right -sticky nsew
            }
        }  
        
        ttk::style layout Card.TFrame {
            Card.field {
                Card.padding -expand 1 
            }
        }

        ttk::style layout TLabelframe {
            Labelframe.border {
                Labelframe.padding -expand 1 -children {
                    Labelframe.label -side left
                }
            }
        }

        ttk::style layout TNotebook {
            Notebook.border -children {
                TNotebook.Tab -expand 1
                Notebook.client -sticky nsew
            }
        }

        ttk::style layout Treeview.Item {
            Treeitem.padding -sticky nsew -children {
                Treeitem.image -side left -sticky {}
                Treeitem.indicator -side left -sticky {}
                Treeitem.text -side left -sticky {}
            }
        }

        # Button
        ttk::style configure TButton -padding {8 4} -anchor center -foreground $colors(-fg)

        ttk::style map TButton -foreground \
            [list disabled #7a7a7a \
                pressed #d0d0d0]

        ttk::style element create Button.button image \
            [list $images(button-rest) \
                {selected disabled} $images(button-disabled) \
                disabled $images(button-disabled) \
                selected $images(button-rest) \
                pressed $images(button-pressed) \
                active $images(button-hover) \
            ] -border 4 -sticky nsew

        # Toolbutton
        ttk::style configure Toolbutton -padding {8 4} -anchor center

        ttk::style element create Toolbutton.button image \
            [list $images(empty) \
                {selected disabled} $images(button-disabled) \
                selected $images(button-rest) \
                pressed $images(button-pressed) \
                active $images(button-hover) \
            ] -border 4 -sticky nsew

        # Menubutton
        ttk::style configure TMenubutton -padding {8 4 0 4}

        ttk::style element create Menubutton.button \
            image [list $images(button-rest) \
                disabled $images(button-disabled) \
                pressed $images(button-pressed) \
                active $images(button-hover) \
            ] -border 4 -sticky nsew 

        ttk::style element create Menubutton.indicator image $images(arrow-down) -width 28 -sticky {}

        # OptionMenu
        ttk::style configure TOptionMenu -padding {8 4 0 4}
        ttk::style configure OptionMenudropdown -borderwidth 0 -relief ridge

        ttk::style element create OptionMenu.button \
            image [list $images(button-rest) \
                disabled $images(button-disabled) \
                pressed $images(button-pressed) \
                active $images(button-hover) \
            ] -border 0 -sticky nsew 

        ttk::style element create OptionMenu.indicator image $images(arrow-down) -width 28 -sticky {}

        # Accent.TButton
        ttk::style configure Accent.TButton -padding {8 4} -anchor center -foreground #ffffff

        ttk::style map Accent.TButton -foreground \
            [list pressed #25536a \
                disabled #a5a5a5]

        ttk::style element create AccentButton.button image \
            [list $images(button-accent-rest) \
                {selected disabled} $images(button-accent-disabled) \
                disabled $images(button-accent-disabled) \
                selected $images(button-accent-rest) \
                pressed $images(button-accent-pressed) \
                active $images(button-accent-hover) \
            ] -border 4 -sticky nsew

        # Titlebar.TButton
        ttk::style configure Titlebar.TButton -padding {8 4} -anchor center -foreground #ffffff

        ttk::style map Titlebar.TButton -foreground \
            [list disabled #6f6f6f \
                pressed #d1d1d1 \
                active #ffffff]

        ttk::style element create TitlebarButton.button image \
            [list $images(empty) \
                disabled $images(empty) \
                pressed $images(button-titlebar-pressed) \
                active $images(button-titlebar-hover) \
            ] -border 4 -sticky nsew

        # Close.Titlebar.TButton
        ttk::style configure Close.Titlebar.TButton -padding {8 4} -anchor center -foreground #ffffff

        ttk::style map Close.Titlebar.TButton -foreground \
            [list disabled #6f6f6f \
                pressed #e8bfbb \
                active #ffffff]

        ttk::style element create CloseButton.button image \
            [list $images(empty) \
                disabled $images(empty) \
                pressed $images(button-close-pressed) \
                active $images(button-close-hover) \
            ] -border 4 -sticky nsew

        # Checkbutton
        ttk::style configure TCheckbutton -padding 2

        ttk::style element create Checkbutton.indicator image \
            [list $images(check-unsel-rest) \
                {alternate disabled} $images(check-tri-disabled) \
                {selected disabled} $images(check-disabled) \
                disabled $images(check-unsel-disabled) \
                {pressed alternate} $images(check-tri-hover) \
                {active alternate} $images(check-tri-hover) \
                alternate $images(check-tri-rest) \
                {pressed selected} $images(check-hover) \
                {active selected} $images(check-hover) \
                selected $images(check-rest) \
                {pressed !selected} $images(check-unsel-pressed) \
                active $images(check-unsel-hover) \
            ] -width 26 -sticky w

        # Switch.TCheckbutton
        ttk::style element create Switch.indicator image \
            [list $images(switch-off-rest) \
                {selected disabled} $images(switch-on-disabled) \
                disabled $images(switch-off-disabled) \
                {pressed selected} $images(switch-on-pressed) \
                {active selected} $images(switch-on-hover) \
                selected $images(switch-on-rest) \
                {pressed !selected} $images(switch-off-pressed) \
                active $images(switch-off-hover) \
            ] -width 46 -sticky w

        # Toggle.TButton
        ttk::style configure Toggle.TButton -padding {8 4 8 4} -anchor center -foreground $colors(-fg)

        ttk::style map Toggle.TButton -foreground \
            [list {selected disabled} #a5a5a5 \
                {selected pressed} #d0d0d0 \
                selected #ffffff \
                pressed #25536a \
                disabled #7a7a7a
            ]


        ttk::style element create ToggleButton.button image \
            [list $images(button-rest) \
                {selected disabled} $images(button-accent-disabled) \
                disabled $images(button-disabled) \
                {pressed selected} $images(button-rest) \
                {active selected} $images(button-accent-hover) \
                selected $images(button-accent-rest) \
                {pressed !selected} $images(button-accent-rest) \
                active $images(button-hover) \
            ] -border 4 -sticky nsew

        # Radiobutton
        ttk::style configure TRadiobutton -padding 0

        ttk::style element create Radiobutton.indicator image \
            [list $images(radio-unsel-rest) \
                {selected disabled} $images(radio-disabled) \
                disabled $images(radio-unsel-disabled) \
                {pressed selected} $images(radio-pressed) \
                {active selected} $images(radio-hover) \
                selected $images(radio-rest) \
                {pressed !selected} $images(radio-unsel-pressed) \
                active $images(radio-unsel-hover) \
            ] -width 20 -sticky w

        ttk::style configure Menu.TRadiobutton -padding 0

        ttk::style element create Menu.Radiobutton.indicator image \
            [list $images(radio-unsel-rest) \
                {selected disabled} $images(radio-disabled) \
                disabled $images(radio-unsel-disabled) \
                {pressed selected} $images(radio-pressed) \
                {active selected} $images(radio-hover) \
                selected $images(radio-rest) \
                {pressed !selected} $images(radio-unsel-pressed) \
                active $images(radio-unsel-hover) \
            ] -width 20 -sticky w

        # Scrollbar

        #ttk::style layout Vertical.TScrollbar

        ttk::style element create Horizontal.Scrollbar.trough image $images(scroll-hor-trough) -sticky ew -border 0
        ttk::style element create Horizontal.Scrollbar.thumb image $images(scroll-hor-thumb) -sticky ew -border 3

        ttk::style element create Horizontal.Scrollbar.rightarrow image $images(scroll-right) -sticky {} -width 13
        ttk::style element create Horizontal.Scrollbar.leftarrow image $images(scroll-left) -sticky {} -width 13

        ttk::style element create Vertical.Scrollbar.trough image $images(scroll-vert-trough) -sticky ns -border 0
        ttk::style element create Vertical.Scrollbar.thumb image $images(scroll-vert-thumb) -sticky ns -border 3

        ttk::style element create Vertical.Scrollbar.uparrow image $images(scroll-up) -sticky {} -height 13
        ttk::style element create Vertical.Scrollbar.downarrow image $images(scroll-down) -sticky {} -height 13

        # Scale
        ttk::style element create Horizontal.Scale.trough image $images(scale-trough-hor) \
            -border 5 -padding 0

        ttk::style element create Vertical.Scale.trough image $images(scale-trough-vert) \
            -border 5 -padding 0

        ttk::style element create Scale.slider \
            image [list $images(scale-thumb-rest) \
                disabled $images(scale-thumb-disabled) \
                pressed $images(scale-thumb-pressed) \
                active $images(scale-thumb-hover) \
            ] -sticky {}

        # Progressbar
        ttk::style element create Horizontal.Progressbar.trough image $images(progress-trough-hor) \
            -border 1 -sticky ew

        ttk::style element create Horizontal.Progressbar.pbar image $images(progress-pbar-hor) \
            -border 2 -sticky ew

        ttk::style element create Vertical.Progressbar.trough image $images(progress-trough-vert) \
            -border 1 -sticky ns

        ttk::style element create Vertical.Progressbar.pbar image $images(progress-pbar-vert) \
            -border 2 -sticky ns

        # Entry
        ttk::style configure TEntry -foreground $colors(-fg)

        ttk::style map TEntry -foreground \
            [list disabled #757575 \
                pressed #cfcfcf
            ]

        ttk::style element create Entry.field \
            image [list $images(entry-rest) \
                {focus hover !invalid} $images(entry-focus) \
                invalid $images(entry-invalid) \
                disabled $images(entry-disabled) \
                {focus !invalid} $images(entry-focus) \
                hover $images(entry-hover) \
            ] -border 5 -padding 8 -sticky nsew

        # Combobox
        ttk::style configure TCombobox -foreground $colors(-fg)

        ttk::style map TCombobox -foreground \
            [list disabled #757575 \
                pressed #cfcfcf
            ]

        ttk::style configure TCombobox -foreground $colors(-fg)
        ttk::style configure ComboboxPopdownFrame -borderwidth 3 -relief solid

        ttk::style map TCombobox -selectbackground [list \
            {readonly hover} $colors(-selectbg) \
            {readonly focus} $colors(-selectbg) \
        ] -selectforeground [list \
            {readonly hover} $colors(-selectfg) \
            {readonly focus} $colors(-selectfg) \
        ]

        ttk::style element create Combobox.field \
            image [list $images(button-rest) \
                {readonly disabled} $images(button-disabled) \
                {readonly pressed} $images(button-rest) \
                {readonly hover} $images(button-hover) \
                readonly $images(button-rest) \
                invalid $images(entry-invalid) \
                disabled $images(combo-disabled) \
                focus $images(entry-focus) \
                hover $images(button-hover) \
            ] -border 5 -padding 8 -sticky nsew
            
        ttk::style element create Combobox.arrow image $images(arrow-down) -width 35 -sticky {}

        # Spinbox
        ttk::style configure TSpinbox -foreground $colors(-fg)

        ttk::style map TSpinbox -foreground \
            [list disabled #757575 \
                pressed #cfcfcf
            ]

        ttk::style element create Spinbox.field \
            image [list $images(entry-rest) \
                invalid $images(entry-invalid) \
                disabled $images(entry-disabled) \
                focus $images(entry-focus) \
                hover $images(entry-hover) \
            ] -border 5 -padding {8 8 54 8} -sticky nsew

        ttk::style element create Spinbox.uparrow image $images(arrow-up) -width 35 -sticky {}
        ttk::style element create Spinbox.downarrow image $images(arrow-down) -width 35 -sticky {}

        # Sizegrip
        ttk::style element create Sizegrip.sizegrip image $images(sizegrip) \
            -sticky nsew

        # Separator
        ttk::style element create TSeparator.separator image $images(separator)

        # Card
        ttk::style element create Card.field image $images(card) \
            -border 10 -padding 4 -sticky nsew

        # Labelframe
        ttk::style element create Labelframe.border image $images(card) \
            -border 5 -padding 4 -sticky nsew
        
        # Notebook
        ttk::style configure TNotebook -padding 1

        ttk::style element create Notebook.border \
            image $images(notebook-border) -border 5 -padding 5

        ttk::style element create Notebook.client image $images(notebook)

        ttk::style element create Notebook.tab \
            image [list $images(tab-rest) \
                selected $images(tab-selected) \
                active $images(tab-hover) \
            ] -border 13 -padding {16 14 16 6} -height 32

        # Treeview
        ttk::style element create Treeview.field image $images(card) \
            -border 5

        ttk::style element create Treeheading.cell \
            image [list $images(treeheading-rest) \
                pressed $images(treeheading-pressed) \
                active $images(treeheading-hover)
            ] -border 5 -padding 15 -sticky nsew
        
        ttk::style element create Treeitem.indicator \
            image [list $images(arrow-right) \
                user2 $images(empty) \
                user1 $images(arrow-down) \
            ] -width 26 -sticky {}

        ttk::style configure Treeview -background $colors(-bg) -rowheight [expr {[font metrics font -linespace] + 2}]
        ttk::style map Treeview \
            -background [list selected #292929] \
            -foreground [list selected $colors(-selectfg)]

        # Panedwindow
        # Insane hack to remove clam's ugly sash
        ttk::style configure Sash -gripcount 0
    }
}