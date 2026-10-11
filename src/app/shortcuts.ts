/** Keyboard shortcuts of the canvas. The dialog (?) lists them; MathCanvas handles them. */
export interface Shortcut {
  keys: string[];
  label: string;
  group: 'Edit' | 'Tools' | 'Recognition' | 'View';
}

export const SHORTCUTS: Shortcut[] = [
  { group: 'Edit', keys: ['Ctrl', 'Z'], label: 'Undo' },
  { group: 'Edit', keys: ['Ctrl', 'Shift', 'Z'], label: 'Redo' },
  { group: 'Edit', keys: ['Ctrl', 'Y'], label: 'Redo' },
  { group: 'Edit', keys: ['Ctrl', 'A'], label: 'Select all ink' },
  { group: 'Edit', keys: ['Ctrl', 'V'], label: 'Paste an image of an exercise onto the page' },
  { group: 'Edit', keys: ['Del'], label: 'Delete the selection (ink or pasted image)' },
  { group: 'Edit', keys: ['Esc'], label: 'Clear selection / close panel' },
  { group: 'Tools', keys: ['P'], label: 'Pen' },
  { group: 'Tools', keys: ['M'], label: 'Highlighter (marker)' },
  { group: 'Tools', keys: ['E'], label: 'Eraser' },
  { group: 'Tools', keys: ['L'], label: 'Lasso: select, then drag to move' },
  { group: 'Tools', keys: ['H'], label: 'Hand: move around the page' },
  { group: 'Tools', keys: ['Space'], label: 'Hold to move around the page' },
  { group: 'Tools', keys: ['1', '2', '3'], label: 'Fine, medium, bold pen' },
  { group: 'Recognition', keys: ['Ctrl', 'Enter'], label: 'Recognize and solve' },
  { group: 'Recognition', keys: ['Enter'], label: 'Recognize and solve' },
  { group: 'Recognition', keys: ['Ctrl', 'K'], label: 'Symbol library' },
  { group: 'View', keys: ['Ctrl', '+'], label: 'Zoom in' },
  { group: 'View', keys: ['Ctrl', '-'], label: 'Zoom out' },
  { group: 'View', keys: ['Ctrl', '0'], label: 'Zoom 100%' },
  { group: 'View', keys: ['Home'], label: 'Back to the start of the page' },
  { group: 'View', keys: ['Ctrl', 'Wheel'], label: 'Zoom with the mouse wheel' },
  { group: 'View', keys: ['Ctrl', 'S'], label: 'Save the page as an image' },
  { group: 'View', keys: ['?'], label: 'This list' },
];
