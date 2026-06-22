import { Tooltip, IconButton } from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

// Small ℹ️ that shows the desktop app's help text. Multi-line text is preserved.
export function HelpTip({ text }: { text?: string }) {
  if (!text) return null;
  return (
    <Tooltip
      title={<span style={{ whiteSpace: "pre-line" }}>{text}</span>}
      enterTouchDelay={0}
      leaveTouchDelay={6000}
      arrow
    >
      <IconButton size="small" sx={{ p: 0.25, color: "text.secondary" }} tabIndex={-1}>
        <InfoOutlinedIcon sx={{ fontSize: 16 }} />
      </IconButton>
    </Tooltip>
  );
}
