# VS Code

## Shortcuts

### Focusing Specific Editor Groups

- To change focus between split editor windows in VS Code, the following shortcuts can be used:
  - `Ctrl + 1`: Focuses the left editor group
  - `Ctrl + 2`: Focuses the right side editor group (if two are open)
  - `Ctrl + 3`: Focuses the right most editor group (if three are open)

### Show/Hide Sidebar

- To show or hide sidebars being shown in the vs code. There are two sidebars like one on the left and one on the right.
  - Chat Window in the left Side: 
    - Shortcut: Ctrl + B
  - Chat Window in Right Side:
    - Shortcut: Ctrl + Alt + B

Visual Studio Code offers several methods for editing or typing on multiple lines simultaneously, which is known as multi-cursor editing.

1. Adding Cursors on Consecutive Lines:
Windows: Press Ctrl + Alt + Down Arrow or Ctrl + Alt + Up Arrow to add a cursor to the line below or above the current cursor.

**_to_study_**

## How to

### Keyboard Output Messed Up

- "£" sigh coming instead of Hashtag on pressing `Shift + 3`
- > By mistake you would have pressed `Alt + Shift` leading to keyboard input changing to UK from US. This will cause a lot of key mapping to get changed like @ will exchange location with " Press `Alt + Shift` it again and it would work.

## Extensions

1. Markdown All in One: Easing the markdown creation
   1. Features:
   2. List Editing: Easy to sequence, edit change List in between
   3. Toogle Checkbox: Check Uncheck Checkbox on clicking Alt + C
2. mardownlint: Suggestions to write properly formatted codes and not too clusmy and unredable markdown
   1. Fix those which can be easily fixed with a single click. `Ctrl + .` and choose fix all
   2. Prettier doesn't give warning for Markdown voilations, even though on using Prettier to reformat solves the issues. Hence to have an eye and check warnings are there, this is also installed along with the Prettier.
3. Prettier: for consistent code formatting
   1. Shortcut for applying it in vs code windows is `Alt + Shift + F`
4. Code Spell Checker: Spelling checker for source code
   1. Since I was doing a lot english word mistakes in writing notes and whatever i am writing hence needed something to suggest the correct spelling and also keep an eye on the same. 
5. vscode-pdf: Display pdf file in VSCode.

## Modifications

1. Markdown All in One: Shortcut to Bold:
   1. Condition: The Ctrl + B is by default the key to hide and show in the vs code, but with this extension, it is acting for B of the text, which is not allowing it to hide/show left sidebar. To resolve it, I am doing the change in the keyboard shortcut of the extension for the keybinding of the Ctrl + B, by changing the When condition of the Keybinding of this extension. I am adding that it works only when text is selected:
      1. By Deafult When: editorTextFocus && !editorReadonly && editorLangId =~ /^markdown$|^rmd$|^quarto$/
      2. New When: editorHasSelection && editorTextFocus && !editorReadonly && editorLangId =~ /^markdown$|^rmd$|^quarto$/
   2. Output: The desired output is achieved.