import { AnnotateSidebar, type AnnotateSidebarProps } from "./AnnotateSidebar";

interface MobileAnnotateSheetProps extends AnnotateSidebarProps {
  open: boolean;
  onClose: () => void;
}

export function MobileAnnotateSheet({ open, onClose, ...sidebarProps }: MobileAnnotateSheetProps) {
  if (!open) return null;

  return (
    <div className="mobile-sheet" role="dialog" aria-modal="true" aria-label="Annotation menu">
      <button type="button" className="mobile-sheet__backdrop" onClick={onClose} aria-label="Close menu" />
      <div className="mobile-sheet__panel">
        <div className="mobile-sheet__header">
          <h2>Annotation menu</h2>
          <button type="button" className="toolbar-btn" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="mobile-sheet__body">
          <AnnotateSidebar {...sidebarProps} compact />
        </div>
      </div>
    </div>
  );
}
