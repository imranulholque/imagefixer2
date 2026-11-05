IMAGE FIXER  (Windows)
======================
Purpose: take many photos of papers (WhatsApp pictures, phone photos, scans), make them PRINT-READY in a few clicks
and drag them straight into your print program.

QUICK WORKFLOW
 1. Add pictures: button, drag & drop, Ctrl+V, or "Add from phone".
 2. Press  "Auto clean ALL pictures"  (right side). Shadows, grey paper and green tint turn white, text gets clear.
 3. Slide through the big pictures with the arrows / Left-Right keys. Fine tune with the sliders if you like.
 4. Save all  -> pictures or one PDF.    Or select thumbnails and drag them out ("Drag out" button / the thumbnails).

THE SLIDERS (right side, most important first)
 Remove shadow and grey / green tint ... evens out the light on the paper; text, stamps and shapes stay
 Whiter paper ........................ light grey paper becomes pure white (saves ink)
 Sharper text and lines .............. clearer thin text and shapes
 Darker, bolder text ................. faint writing gets stronger, paper stays white
 Colour strength ..................... more saturated colours (black and white stay as they are)
 Double-click a slider to reset it.   "Use for all pictures" copies the settings to every picture.

TEXT COLOURS
 "Pick a colour from the picture" -> click on some text. That colour gets its own row:
  Strength slider: left = fades into the paper, middle = same, right = stronger / darker.
  Change colour: turns that ink into another colour.

TOOLS (above the picture)
 Move view ... drag the picture; mouse wheel zooms; double-click fits; Shift + wheel = next picture
 Pick color ... see above
 Erase ........ Brush / Box / Free shape. The part is gone when you let go (Ctrl+Z to undo). The hole is filled like
                the paper around it - good for removing parts of a form.
 Select / Move  Draw an area (Box / Free shape / Brush). Drawing more ADDS to it; "Subtract" (or Alt) takes a part
                out; Ctrl+Z undoes the last selection step; "Clear selection" / Esc / Ctrl+D removes it.
                Then drag inside the area to move it. "Copy" keeps the original in place.
                "Erase selected" (or Delete) removes the area.
 Crop ......... drag the box, Enter applies, Esc cancels
 Rotate left / right, Angle (any angle), Straighten page (finds the paper in a photo), Undo, Fit
 Hold: original ... press and hold to compare with the picture before editing

SAVING AND DRAGGING OUT
 "Save selected" / "Save all": as pictures (JPG keeps the original format), one PDF with A4 pages, or one PDF whose
 pages have the picture's shape. Page order = order of the thumbnails (Earlier / Later buttons).
 Drag out: select one or many thumbnails (Ctrl / Shift + click), then drag them (or the "Drag out" button) into Explorer,
 a print program, WhatsApp Web ... The edited full-size file is dragged.

PHONE AS A SCANNER ("Add from phone")
 Scan the QR code with the phone (same Wi-Fi). The address is found automatically. You can set the PORT in that window
 ("Use this port"). A web page opens in the phone - no app needed:
  Choose file: send pictures.   Scan: photograph a paper, it is straightened automatically, drag the 4 corners if needed.
 The link has a secret code that changes when you press "New code". The first time, Windows may ask to allow the program
 on private networks - allow it.

HOW TO GET THE PROGRAM
 1. Make a new GitHub repository and upload everything in this folder (keep the .github folder and icon.ico).
 2. Actions tab -> "Build" -> latest run -> Artifacts -> ImageFixer -> download, unzip.
 3. Put the folder somewhere permanent (for example C:\Tools\ImageFixer) and run ImageFixer.exe.
 Explorer: select pictures -> right-click -> Send to -> Image Fixer.

KEYS
 Left / Right: previous / next      Ctrl+Z: undo      Ctrl+V: paste      Ctrl+S: save selected
 Ctrl+Shift+S: save all             Ctrl+0: fit       Delete: erase selected area   Enter / Esc: apply / cancel crop

SETTINGS (the "..." button): start with Windows, Send to entry, save folder, phone port.
Closing the window keeps the program in the tray; tray icon -> Exit quits.
Problems are written to  %APPDATA%\ImageFixer\error.log
